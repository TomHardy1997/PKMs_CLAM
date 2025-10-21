# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm

def get_magnification_from_wsi(wsi_object, slide_path=None, save_unreadable=False, log_file=None):
    """
    从WSI文件中提取放大倍率信息
    """
    try:
        wsi = wsi_object.getOpenSlide()
        
        # 方法1: 从properties中直接读取
        if 'aperio.AppMag' in wsi.properties:
            mag = float(wsi.properties['aperio.AppMag'])
            print(f"Found Aperio magnification: {mag}X")
            return mag
        
        # 方法2: 从objective power读取
        elif 'openslide.objective-power' in wsi.properties:
            mag = float(wsi.properties['openslide.objective-power'])
            print(f"Found OpenSlide objective power: {mag}X")
            return mag
        
        # 方法3: 从MPP (microns per pixel) 计算
        elif 'openslide.mpp-x' in wsi.properties:
            mpp_x = float(wsi.properties['openslide.mpp-x'])
            # 40X通常对应0.25 mpp, 20X对应0.5 mpp
            if mpp_x <= 0.3:
                mag = 40.0
            elif mpp_x <= 0.6:
                mag = 20.0
            else:
                mag = 10.0
            print(f"Estimated magnification from MPP ({mpp_x}): {mag}X")
            return mag
        
        # 方法4: 其他vendor特定的属性
        elif 'tiff.ImageDescription' in wsi.properties:
            desc = wsi.properties['tiff.ImageDescription']
            if '40' in desc and ('x' in desc.lower() or 'X' in desc):
                return 40.0
            elif '20' in desc and ('x' in desc.lower() or 'X' in desc):
                return 20.0
        
        # 无法读取放大倍数时的处理
        print("Warning: Could not determine magnification, defaulting to 20X")
        if save_unreadable and slide_path and log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{slide_path}\n")
            print(f"Saved unreadable slide path to: {log_file}")
        
        return 20.0
        
    except Exception as e:
        print(f"Error reading magnification: {e}")
        print("Defaulting to 20X magnification")
        
        # 保存无法读取的文件路径
        if save_unreadable and slide_path and log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{slide_path} (Error: {str(e)})\n")
            print(f"Saved error slide path to: {log_file}")
        
        return 20.0


def get_patch_size_by_magnification(magnification, base_size_20x=512):
    """
    根据放大倍率确定patch_size和step_size
    
    Args:
        magnification: WSI的放大倍率
        base_size_20x: 20X下的基础patch大小 (默认512)
    
    Returns:
        patch_size, step_size: 调整后的patch和step大小
    """
    if magnification >= 40:
        # 40X: 使用更大的patch保证相同物理尺寸
        patch_size = base_size_20x * 2  # 1024
        step_size = patch_size  # 1024
        print(f"40X magnification detected -> patch_size: {patch_size}, step_size: {step_size}")
    elif magnification >= 20:
        # 20X: 使用基础大小
        patch_size = base_size_20x  # 512
        step_size = patch_size  # 512
        print(f"20X magnification detected -> patch_size: {patch_size}, step_size: {step_size}")
    else:
        # 10X或更低: 使用更小的patch
        patch_size = base_size_20x // 2  # 256
        step_size = patch_size  # 256
        print(f"{magnification}X magnification detected -> patch_size: {patch_size}, step_size: {step_size}")
    
    return patch_size, step_size


def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  auto_adjust_patch_size = True,
				  base_patch_size_20x = 512,  # 新增：20X下的基础patch大小
				  save_unreadable_paths = False,  # 新增：是否保存无法读取放大倍数的文件路径
				  unreadable_log_file = 'unreadable_magnification.txt',  # 新增：日志文件名
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	
	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	# 添加新列来记录放大倍率和调整后的patch信息
	if 'magnification' not in df.columns:
		df['magnification'] = 0.0
	if 'adjusted_patch_size' not in df.columns:
		df['adjusted_patch_size'] = 0
	if 'adjusted_step_size' not in df.columns:
		df['adjusted_step_size'] = 0

	# 初始化日志文件
	if save_unreadable_paths:
		log_file_path = os.path.join(save_dir, unreadable_log_file)
		# 清空之前的日志文件
		with open(log_file_path, 'w', encoding='utf-8') as f:
			f.write("# Slides with unreadable magnification:\n")
		print(f"Unreadable magnification log will be saved to: {log_file_path}")
	else:
		log_file_path = None

	mask = df['process'] == 1
	process_stack = df[mask]
	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)

		# 检测放大倍率并调整patch_size
		current_magnification = get_magnification_from_wsi(
			WSI_object, 
			slide_path=full_path,
			save_unreadable=save_unreadable_paths,
			log_file=log_file_path
		)
		df.loc[idx, 'magnification'] = current_magnification

		if auto_adjust_patch_size:
			adjusted_patch_size, adjusted_step_size = get_patch_size_by_magnification(
				current_magnification, base_patch_size_20x
			)
			df.loc[idx, 'adjusted_patch_size'] = adjusted_patch_size
			df.loc[idx, 'adjusted_step_size'] = adjusted_step_size
			
			print(f"Slide: {slide}")
			print(f"Magnification: {current_magnification}X")
			print(f"Using patch_size: {adjusted_patch_size}, step_size: {adjusted_step_size}")
		else:
			adjusted_patch_size = patch_size
			adjusted_step_size = step_size
			df.loc[idx, 'adjusted_patch_size'] = adjusted_patch_size
			df.loc[idx, 'adjusted_step_size'] = adjusted_step_size
			print(f"Auto-adjust disabled, using original patch_size: {patch_size}, step_size: {step_size}")

		# 其余代码保持不变...
		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1
		if patch:
			# 使用调整后的patch_size和step_size
			current_patch_params.update({
				'patch_level': patch_level, 
				'patch_size': adjusted_patch_size,    # 使用调整后的大小
				'step_size': adjusted_step_size,      # 使用调整后的步长
				'save_path': patch_save_dir
			})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object, **current_patch_params)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
	
	# 输出无法读取放大倍数的文件统计
	if save_unreadable_paths and os.path.exists(log_file_path):
		with open(log_file_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		unreadable_count = len([line for line in lines if not line.startswith('#') and line.strip()])
		print(f"\nTotal slides with unreadable magnification: {unreadable_count}")
		if unreadable_count > 0:
			print(f"Check details in: {log_file_path}")
		
	return seg_times, patch_times

# 参数解析部分
parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--auto_adjust_patch_size', default=True, action='store_true',
                    help='automatically adjust patch size based on magnification')
parser.add_argument('--no_auto_adjust_patch_size', dest='auto_adjust_patch_size', 
                    action='store_false',
                    help='disable automatic patch size adjustment')
parser.add_argument('--base_patch_size_20x', type=int, default=512,
                    help='base patch size for 20X magnification (default: 512)')
parser.add_argument('--save_unreadable_paths', default=False, action='store_true',
                    help='save full paths of slides with unreadable magnification to a file')
parser.add_argument('--unreadable_log_file', type=str, default='unreadable_magnification.txt',
                    help='filename to save paths of slides with unreadable magnification')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)
	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, 
											step_size = args.step_size, 
											auto_adjust_patch_size = args.auto_adjust_patch_size,
											base_patch_size_20x = args.base_patch_size_20x,
											save_unreadable_paths = args.save_unreadable_paths,
											unreadable_log_file = args.unreadable_log_file,
											seg = args.seg, 
											use_default_params = False, 
											save_mask = True, 
											stitch = args.stitch,
											patch_level = args.patch_level, 
											patch = args.patch,
											process_list = process_list, 
											auto_skip = args.no_auto_skip)
