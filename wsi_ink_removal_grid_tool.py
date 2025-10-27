"""
WSI Grid Selector - Batch Processing Version
支持批量处理、进度管理、结果查看
"""

import os
import cv2
import h5py
import json
import glob
import numpy as np
from PIL import Image
import openslide
from datetime import datetime


class WSIGridSelector:
    """
    Interactive WSI grid selector with batch processing support.
    """
    
    def __init__(self, svs_file_path, output_dir='.', scale_factor=100):
        self.svs_file_path = svs_file_path
        self.file_basename = os.path.splitext(os.path.basename(svs_file_path))[0]
        
        self.output_dir = os.path.join(output_dir, self.file_basename)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scale_factor = scale_factor
        self.slide = None
        self.thumbnail_image = None
        self.clone = None
        
        # 绿色多边形（包含区域）
        self.points = []
        self.green_polygons = []  # 存储所有绿色多边形
        
        # 蓝色多边形（排除区域）
        self.blue_points = []
        self.blue_polygons = []  # 存储所有蓝色多边形
        
        self.grid_coordinates = []
        self.rect_size = 512
        self.is_completed = False
        
        # 当前模式：'green' 或 'blue'
        self.current_mode = 'green'

        
    def load_slide(self):
        """Load the WSI slide using OpenSlide."""
        try:
            self.slide = openslide.OpenSlide(self.svs_file_path)
            
            objective_power = self.slide.properties.get(
                openslide.PROPERTY_NAME_OBJECTIVE_POWER, '20'
            )
            
            if float(objective_power) == 40:
                self.rect_size = 1024
            elif float(objective_power) == 20:
                self.rect_size = 512
            else:
                self.rect_size = 512
                
            print(f"✓ Slide loaded: {self.file_basename}")
            print(f"  - Dimensions: {self.slide.dimensions}")
            print(f"  - Objective power: {objective_power}x")
            print(f"  - Grid size: {self.rect_size}px")
            return True
            
        except Exception as e:
            print(f"✗ Error loading slide: {e}")
            return False
    
    def generate_thumbnail(self):
        """Generate thumbnail using OpenSlide."""
        try:
            width, height = self.slide.dimensions
            thumb_width = width // self.scale_factor
            thumb_height = height // self.scale_factor
            
            thumbnail = self.slide.get_thumbnail((thumb_width, thumb_height))
            
            # 保存缩略图（使用文件名）
            thumbnail_filename = f"{self.file_basename}_thumbnail.png"
            thumbnail_path = os.path.join(self.output_dir, thumbnail_filename)
            thumbnail.save(thumbnail_path)
            
            self.thumbnail_image = cv2.cvtColor(
                np.array(thumbnail), cv2.COLOR_RGB2BGR
            )
            self.clone = self.thumbnail_image.copy()
            
            print(f"✓ Thumbnail generated: {thumbnail_filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error generating thumbnail: {e}")
            return False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'green':
                self.points.append((x, y))
                color = (0, 255, 0)
                points_list = self.points
            else:
                self.blue_points.append((x, y))
                color = (255, 0, 0)
                points_list = self.blue_points
            
            cv2.circle(self.clone, (x, y), 3, color, -1)
            
            if len(points_list) > 1:
                cv2.line(self.clone, points_list[-2], points_list[-1], color, 2)
            
            cv2.imshow("WSI Grid Selector", self.clone)
            mode_name = "GREEN (Include)" if self.current_mode == 'green' else "BLUE (Exclude)"
            print(f"  [{mode_name}] Point {len(points_list)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_mode == 'green':
                points_list = self.points
                color = (0, 255, 0)
                mode_name = "GREEN"
            else:
                points_list = self.blue_points
                color = (255, 0, 0)
                mode_name = "BLUE"
            
            if len(points_list) > 2:
                cv2.line(self.clone, points_list[-1], points_list[0], color, 2)
                
                if self.current_mode == 'green':
                    self.green_polygons.append(list(self.points))
                    polygon_count = len(self.green_polygons)
                    print(f"✓ {mode_name} polygon #{polygon_count} completed with {len(self.points)} vertices")
                    print(f"  📊 Total GREEN polygons: {polygon_count}")
                    self.points = []
                else:
                    self.blue_polygons.append(list(self.blue_points))
                    zone_count = len(self.blue_polygons)
                    print(f"✓ {mode_name} exclusion zone #{zone_count} completed with {len(self.blue_points)} vertices")
                    print(f"  📊 Total BLUE zones: {zone_count}")
                    self.blue_points = []
                
                cv2.imshow("WSI Grid Selector", self.clone)
            else:
                print("⚠️  Need at least 3 points to form a polygon")

    
    def _fill_rectangles_in_polygon(self):
        """Generate grid within green polygons, excluding blue polygons."""
        width, height = self.slide.dimensions
        thumb_height, thumb_width = self.thumbnail_image.shape[:2]
        
        print(f"\n[DEBUG] Generating grid...")
        print(f"  - Slide dimensions: {width} x {height}")
        print(f"  - Thumbnail dimensions: {thumb_width} x {thumb_height}")
        print(f"  - Grid size: {self.rect_size}px")
        print(f"  - Scale factor: {self.scale_factor}")
        
        # 创建缩略图尺寸的mask
        green_mask = np.zeros((thumb_height, thumb_width), dtype=np.uint8)
        blue_mask = np.zeros((thumb_height, thumb_width), dtype=np.uint8)
        
        # 填充多边形
        for polygon in self.green_polygons:
            cv2.fillPoly(green_mask, [np.array(polygon, dtype=np.int32)], 255)
        
        for polygon in self.blue_polygons:
            cv2.fillPoly(blue_mask, [np.array(polygon, dtype=np.int32)], 255)
        
        print(f"  - Green area: {np.sum(green_mask == 255):,} pixels")
        print(f"  - Blue area: {np.sum(blue_mask == 255):,} pixels")
        
        # ✅ 在原始坐标系统上遍历（精确！）
        count = 0
        excluded_count = 0
        
        print(f"\n  Generating grids...")
        total_possible = ((width // self.rect_size) * (height // self.rect_size))
        print(f"  - Max possible grids: {total_possible:,}")
        
        for orig_y in range(0, height - self.rect_size, self.rect_size):
            for orig_x in range(0, width - self.rect_size, self.rect_size):
                # ✅ 计算在缩略图上的中心点位置
                thumb_center_x = int((orig_x + self.rect_size // 2) / self.scale_factor)
                thumb_center_y = int((orig_y + self.rect_size // 2) / self.scale_factor)
                
                # 边界检查
                if thumb_center_y >= thumb_height or thumb_center_x >= thumb_width:
                    continue
                
                # 检查是否在mask内
                in_green = green_mask[thumb_center_y, thumb_center_x] == 255
                in_blue = blue_mask[thumb_center_y, thumb_center_x] == 255
                
                if in_green and not in_blue:
                    # ✅ 直接使用原始坐标（已经是精确的rect_size倍数）
                    self.grid_coordinates.append((
                        (orig_x, orig_y),
                        (orig_x + self.rect_size, orig_y + self.rect_size)
                    ))
                    
                    # 在缩略图上画框
                    thumb_x1 = int(orig_x / self.scale_factor)
                    thumb_y1 = int(orig_y / self.scale_factor)
                    thumb_x2 = int((orig_x + self.rect_size) / self.scale_factor)
                    thumb_y2 = int((orig_y + self.rect_size) / self.scale_factor)
                    
                    cv2.rectangle(
                        self.clone,
                        (thumb_x1, thumb_y1),
                        (thumb_x2, thumb_y2),
                        (0, 0, 255), 1
                    )
                    count += 1
                elif in_green and in_blue:
                    excluded_count += 1
        
        print(f"\n✓ Grid generation completed:")
        print(f"  - Generated: {count:,} grids")
        print(f"  - Excluded by blue zones: {excluded_count:,} grids")
        print(f"  - Coverage: {count * (self.rect_size ** 2):,} px²")


    
    def run_interactive_selection(self):
        """Run interactive selection interface."""
        if self.thumbnail_image is None:
            print("✗ Error: No thumbnail loaded.")
            return False
        
        print("\n" + "="*60)
        print(f"🔬 WSI GRID SELECTOR - {self.file_basename}")
        print("="*60)
        print("📝 Instructions:")
        print("  • Left click: Add polygon vertex")
        print("  • Right click: Complete polygon")
        print("  • Press 'b': Switch to BLUE mode (exclusion zones)")
        print("  • Press 'g': Switch to GREEN mode (inclusion zones)")
        print("  • Press 'z': UNDO last point")
        print("  • Press 'SPACE': Generate grid (after drawing polygons)")
        print("  • Press 'q': Save and go to NEXT file")
        print("  • Press 's': SKIP this file (don't save)")
        print("  • Press 'r': Reset current polygon")
        print("  • Press 'c': Clear all and restart")
        print("  • Press 'ESC': Exit entire program")
        print("="*60)
        print(f"🟢 Current mode: GREEN (Include)")
        print("="*60 + "\n")
        
        cv2.namedWindow("WSI Grid Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("WSI Grid Selector", 1200, 800)
        cv2.setMouseCallback("WSI Grid Selector", self._mouse_callback)
        cv2.imshow("WSI Grid Selector", self.thumbnail_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Saving and moving to next file...")
                self.is_completed = True
                break
            elif key == ord('s'):
                print("\n⚠️  Skipping this file (not saved)")
                self.is_completed = False
                break
            elif key == 27:  # ESC
                print("\n✗ Exiting entire program...")
                cv2.destroyAllWindows()
                return 'exit'
            elif key == ord('b'):  # 切换到蓝色模式
                self.current_mode = 'blue'
                print("\n🔵 Switched to BLUE mode (Exclusion zones)")
                print("   Draw polygons to EXCLUDE from grid generation")
            elif key == ord('g'):  # 切换到绿色模式
                self.current_mode = 'green'
                print("\n🟢 Switched to GREEN mode (Inclusion zones)")
                print("   Draw polygons to INCLUDE in grid generation")
            elif key == ord('z'):  # 撤销
                self._undo_last_point()
            elif key == ord(' '):  # 空格键生成网格
                if self.green_polygons:
                    print("\n🔄 Regenerating grid...")
                    self.grid_coordinates = []
                    
                    # 重置图像
                    self.clone = self.thumbnail_image.copy()
                    
                    # 重绘所有多边形
                    self._redraw_completed_polygons()
                    
                    # 生成新网格
                    self._fill_rectangles_in_polygon()
                    
                    cv2.imshow("WSI Grid Selector", self.clone)
                else:
                    print("\n⚠️  Please draw at least one GREEN polygon first")
                    print("   Hint: Press 'g' for GREEN mode, draw polygon, then press SPACE")
            elif key == ord('r'):
                if self.current_mode == 'green':
                    self.points = []
                else:
                    self.blue_points = []
                self.clone = self.thumbnail_image.copy()
                self._redraw_completed_polygons()
                self._redraw_grids()
                cv2.imshow("WSI Grid Selector", self.clone)
                print("✓ Current polygon reset")
            elif key == ord('c'):
                self.points = []
                self.blue_points = []
                self.green_polygons = []
                self.blue_polygons = []
                self.grid_coordinates = []
                self.clone = self.thumbnail_image.copy()
                cv2.imshow("WSI Grid Selector", self.clone)
                print("✓ All cleared")
        
        cv2.destroyAllWindows()
        return True

    def _redraw_completed_polygons(self):
        """Redraw all completed polygons."""
        # 重绘绿色多边形
        for polygon in self.green_polygons:
            for i in range(len(polygon)):
                # 画点
                cv2.circle(self.clone, polygon[i], 3, (0, 255, 0), -1)
                # 画线
                if i > 0:
                    cv2.line(self.clone, polygon[i-1], polygon[i], (0, 255, 0), 2)
            # 闭合
            if len(polygon) > 2:
                cv2.line(self.clone, polygon[-1], polygon[0], (0, 255, 0), 2)
        
        # 重绘蓝色多边形
        for polygon in self.blue_polygons:
            for i in range(len(polygon)):
                # 画点
                cv2.circle(self.clone, polygon[i], 3, (255, 0, 0), -1)
                # 画线
                if i > 0:
                    cv2.line(self.clone, polygon[i-1], polygon[i], (255, 0, 0), 2)
            # 闭合
            if len(polygon) > 2:
                cv2.line(self.clone, polygon[-1], polygon[0], (255, 0, 0), 2)

    def _redraw_current_polygon(self):
        """Redraw the current polygon being drawn."""
        if self.current_mode == 'green':
            points_list = self.points
            color = (0, 255, 0)
        else:
            points_list = self.blue_points
            color = (255, 0, 0)
        
        for i, point in enumerate(points_list):
            # 画点
            cv2.circle(self.clone, point, 3, color, -1)
            # 画线
            if i > 0:
                cv2.line(self.clone, points_list[i-1], points_list[i], color, 2)

    def _undo_last_point(self):
        """Undo the last added point."""
        if self.current_mode == 'green':
            points_list = self.points
            mode_name = "GREEN"
        else:
            points_list = self.blue_points
            mode_name = "BLUE"
        
        if not points_list:
            print(f"⚠️  No {mode_name} points to undo")
            return
        
        # 移除最后一个点
        removed_point = points_list.pop()
        print(f"↶ Undone {mode_name} point {len(points_list) + 1}: {removed_point}")
        
        # 重绘图像
        self.clone = self.thumbnail_image.copy()
        
        # 重绘已完成的绿色多边形
        self._redraw_completed_polygons()
        
        # 重绘已有的网格
        self._redraw_grids()
        
        # 重绘当前正在画的多边形
        self._redraw_current_polygon()
        
        cv2.imshow("WSI Grid Selector", self.clone)

    def _redraw_grids(self):
        """Redraw existing grids on thumbnail."""
        for (x1, y1), (x2, y2) in self.grid_coordinates:
            scaled_x1 = int(x1 / self.scale_factor)
            scaled_y1 = int(y1 / self.scale_factor)
            scaled_x2 = int(x2 / self.scale_factor)
            scaled_y2 = int(y2 / self.scale_factor)
            cv2.rectangle(self.clone, (scaled_x1, scaled_y1), 
                        (scaled_x2, scaled_y2), (0, 0, 255), 1)

    
    def save_results(self):
        """Save results with unique filenames."""
        if not self.is_completed or len(self.grid_coordinates) == 0:
            print("⚠️  No grids to save")
            return False
        
        try:
            # 保存标注图像（使用文件名）
            output_image = f"{self.file_basename}_annotated.png"
            output_image_path = os.path.join(self.output_dir, output_image)
            cv2.imwrite(output_image_path, self.clone)
            print(f"✓ Annotated image saved: {output_image}")
            
            # 保存坐标（使用文件名）
            coordinates_file = f"{self.file_basename}_coordinates.h5"
            coordinates_path = os.path.join(self.output_dir, coordinates_file)
            with h5py.File(coordinates_path, "w") as h5file:
                coordinates_array = np.array(
                    self.grid_coordinates, dtype=np.int32
                ).reshape(-1, 4)
                
                h5file.create_dataset("coordinates", data=coordinates_array)
                h5file.attrs['scale_factor'] = self.scale_factor
                h5file.attrs['rect_size'] = self.rect_size
                h5file.attrs['total_grids'] = len(self.grid_coordinates)
                h5file.attrs['green_polygons'] = len(self.green_polygons)  # ← 新增
                h5file.attrs['blue_polygons'] = len(self.blue_polygons)    # ← 新增
                h5file.attrs['source_file'] = os.path.basename(self.svs_file_path)
                h5file.attrs['slide_dimensions'] = self.slide.dimensions
                h5file.attrs['processing_date'] = datetime.now().isoformat()
            
            print(f"✓ Coordinates saved: {coordinates_file}")
            print(f"✓ Total grids: {len(self.grid_coordinates)}")
            
            # 保存元数据JSON
            metadata = {
                'source_file': os.path.basename(self.svs_file_path),
                'slide_dimensions': self.slide.dimensions,
                'scale_factor': self.scale_factor,
                'rect_size': self.rect_size,
                'total_grids': len(self.grid_coordinates),
                'green_polygons': len(self.green_polygons),  # ← 新增
                'blue_polygons': len(self.blue_polygons),    # ← 新增
                'processing_date': datetime.now().isoformat(),
                'output_dir': self.output_dir
            }
            
            metadata_file = f"{self.file_basename}_metadata.json"
            metadata_path = os.path.join(self.output_dir, metadata_file)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata saved: {metadata_file}")
            if self.blue_polygons:
                print(f"  (Excluded {len(self.blue_polygons)} blue zones)")
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
            return False
    
    def get_statistics(self):
        """Print statistics."""
        if not self.grid_coordinates:
            return
        
        print("\n" + "="*50)
        print("📊 GRID GENERATION STATISTICS")
        print("="*50)
        print(f"📁 Source: {self.file_basename}")
        print(f"📐 Slide size: {self.slide.dimensions[0]} x {self.slide.dimensions[1]}")
        print(f"🎯 Total grids: {len(self.grid_coordinates)}")
        print(f"📏 Coverage: {len(self.grid_coordinates) * (self.rect_size ** 2):,} px²")
        print("="*50 + "\n")


class BatchProcessor:
    """
    Batch processor for multiple WSI files.
    """
    
    def __init__(self, input_dir, output_dir='./output', scale_factor=100):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.progress_file = os.path.join(output_dir, 'processing_progress.json')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载进度
        self.progress = self._load_progress()
    
    def _load_progress(self):
        """Load processing progress."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'skipped': [], 'pending': []}
    
    def _save_progress(self):
        """Save processing progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_svs_files(self):
        """Get all SVS files in input directory."""
        pattern = os.path.join(self.input_dir, '*.svs')
        files = glob.glob(pattern)
        
        # 也支持其他格式
        for ext in ['*.tif', '*.tiff', '*.ndpi']:
            pattern = os.path.join(self.input_dir, ext)
            files.extend(glob.glob(pattern))
        
        return sorted(files)
    
    def run(self, start_from=None):
        """Run batch processing."""
        svs_files = self.get_svs_files()
        
        if not svs_files:
            print(f"✗ No WSI files found in: {self.input_dir}")
            return
        
        print("\n" + "="*70)
        print("🚀 BATCH PROCESSING MODE")
        print("="*70)
        print(f"📁 Input directory: {self.input_dir}")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"📊 Total files: {len(svs_files)}")
        print(f"✅ Completed: {len(self.progress['completed'])}")
        print(f"⏭️  Skipped: {len(self.progress['skipped'])}")
        print(f"⏳ Pending: {len(svs_files) - len(self.progress['completed']) - len(self.progress['skipped'])}")
        print("="*70 + "\n")
        
        # 确定起始位置
        start_idx = 0
        if start_from:
            for idx, f in enumerate(svs_files):
                if start_from in os.path.basename(f):
                    start_idx = idx
                    break
        
        # 处理每个文件
        for idx, svs_file in enumerate(svs_files[start_idx:], start=start_idx + 1):
            basename = os.path.basename(svs_file)
            
            # 检查是否已处理
            if basename in self.progress['completed']:
                print(f"\n[{idx}/{len(svs_files)}] ✅ Already completed: {basename}")
                continue
            
            if basename in self.progress['skipped']:
                print(f"\n[{idx}/{len(svs_files)}] ⏭️  Previously skipped: {basename}")
                response = input("  Process now? (y/n): ").lower()
                if response != 'y':
                    continue
            
            print(f"\n{'='*70}")
            print(f"📌 Processing [{idx}/{len(svs_files)}]: {basename}")
            print(f"{'='*70}")
            
            # 处理文件
            selector = WSIGridSelector(svs_file, self.output_dir, self.scale_factor)
            
            if not selector.load_slide():
                continue
            
            if not selector.generate_thumbnail():
                continue
            
            result = selector.run_interactive_selection()
            
            if result == 'exit':
                print("\n✗ User requested exit")
                break
            elif result and selector.is_completed:
                selector.save_results()
                selector.get_statistics()
                self.progress['completed'].append(basename)
                self._save_progress()
            else:
                self.progress['skipped'].append(basename)
                self._save_progress()
        
        # 最终统计
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final processing summary."""
        print("\n" + "="*70)
        print("🎉 BATCH PROCESSING COMPLETED")
        print("="*70)
        print(f"✅ Completed: {len(self.progress['completed'])} files")
        print(f"⏭️  Skipped: {len(self.progress['skipped'])} files")
        print(f"📂 Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        if self.progress['completed']:
            print("✅ Completed files:")
            for f in self.progress['completed']:
                print(f"  • {f}")
        
        if self.progress['skipped']:
            print("\n⏭️  Skipped files:")
            for f in self.progress['skipped']:
                print(f"  • {f}")


class ResultViewer:
    """
    Viewer for inspecting processed results.
    """
    
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir
    
    def list_results(self):
        """List all processed results."""
        subdirs = [d for d in os.listdir(self.output_dir) 
                   if os.path.isdir(os.path.join(self.output_dir, d))]
        
        if not subdirs:
            print("⚠️  No results found")
            return []
        
        print("\n" + "="*70)
        print("📋 PROCESSED RESULTS")
        print("="*70)
        
        results = []
        for idx, subdir in enumerate(subdirs, 1):
            subdir_path = os.path.join(self.output_dir, subdir)
            
            # 检查是否有坐标文件
            h5_files = glob.glob(os.path.join(subdir_path, '*_coordinates.h5'))
            if h5_files:
                h5_file = h5_files[0]
                
                with h5py.File(h5_file, 'r') as f:
                    total_grids = f.attrs.get('total_grids', 0)
                    processing_date = f.attrs.get('processing_date', 'Unknown')
                
                results.append({
                    'index': idx,
                    'name': subdir,
                    'path': subdir_path,
                    'h5_file': h5_file,
                    'total_grids': total_grids,
                    'date': processing_date
                })
                
                print(f"[{idx}] {subdir}")
                print(f"    📊 Grids: {total_grids}")
                print(f"    📅 Date: {processing_date}")
        
        print("="*70 + "\n")
        return results
    
    def view_result(self, result_name):
        """View a specific result."""
        result_path = os.path.join(self.output_dir, result_name)
        
        if not os.path.exists(result_path):
            print(f"✗ Result not found: {result_name}")
            return
        
        # 查找标注图像
        annotated_images = glob.glob(os.path.join(result_path, '*_annotated.png'))
        if not annotated_images:
            print(f"✗ No annotated image found")
            return
        
        # 显示图像
        img_path = annotated_images[0]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"✗ Failed to load image")
            return
        
        print(f"\n📷 Viewing: {result_name}")
        print("  Press any key to close...")
        
        cv2.namedWindow(result_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(result_name, 1200, 800)
        cv2.imshow(result_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def extract_coordinates(self, result_name):
        """Extract coordinates from a result."""
        result_path = os.path.join(self.output_dir, result_name)
        h5_files = glob.glob(os.path.join(result_path, '*_coordinates.h5'))
        
        if not h5_files:
            print(f"✗ No coordinates file found")
            return None
        
        h5_file = h5_files[0]
        
        with h5py.File(h5_file, 'r') as f:
            coordinates = f['coordinates'][:]
            attrs = dict(f.attrs)
        
        print(f"\n📊 Coordinates for: {result_name}")
        print(f"  Total grids: {len(coordinates)}")
        print(f"  Grid size: {attrs.get('rect_size', 'Unknown')}")
        print(f"  Slide dimensions: {attrs.get('slide_dimensions', 'Unknown')}")
        
        return coordinates, attrs
    
    def interactive_menu(self):
        """Interactive menu for viewing results."""
        while True:
            results = self.list_results()
            
            if not results:
                break
            
            print("\n📝 Options:")
            print("  • Enter number to view result")
            print("  • Type 'e' + number to extract coordinates (e.g., 'e1')")
            print("  • Type 'q' to quit")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice.startswith('e'):
                try:
                    idx = int(choice[1:])
                    if 1 <= idx <= len(results):
                        coords, attrs = self.extract_coordinates(results[idx-1]['name'])
                        if coords is not None:
                            print(f"\n  Sample coordinates (first 5):")
                            for i, coord in enumerate(coords[:5], 1):
                                print(f"    [{i}] x1={coord[0]}, y1={coord[1]}, x2={coord[2]}, y2={coord[3]}")
                except:
                    print("✗ Invalid input")
            else:
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(results):
                        self.view_result(results[idx-1]['name'])
                except:
                    print("✗ Invalid input")


def main():
    """Main function with menu."""
    print("\n" + "="*70)
    print("🔬 WSI GRID SELECTOR - ENHANCED VERSION")
    print("="*70)
    print("\n📝 Select mode:")
    print("  [1] Batch Processing (process multiple files)")
    print("  [2] Single File Processing")
    print("  [3] View Results (inspect processed files)")
    print("  [4] Exit")
    print("="*70)
    
    choice = input("\nYour choice: ").strip()
    
    if choice == '1':
        # 批量处理模式
        default_input = '/Users/tangdi/Desktop/TCGA_STAD_selected'
        
        input_dir = input(f"\nEnter input directory path (default: {default_input}): ").strip()
        
        # 如果不输入，使用默认路径
        if not input_dir:
            input_dir = default_input
        
        if not os.path.exists(input_dir):
            print(f"✗ Invalid directory: {input_dir}")
            return
        
        output_dir = input("Enter output directory (default: ./output): ").strip()
        if not output_dir:
            output_dir = './output'
        
        processor = BatchProcessor(input_dir, output_dir)
        processor.run()
    
    elif choice == '2':
        # 单文件处理模式
        svs_file = input("\nEnter SVS file path: ").strip()
        if not os.path.exists(svs_file):
            print("✗ File not found")
            return
        
        output_dir = input("Enter output directory (default: ./output): ").strip()
        if not output_dir:
            output_dir = './output'
        
        selector = WSIGridSelector(svs_file, output_dir)
        
        if selector.load_slide() and selector.generate_thumbnail():
            if selector.run_interactive_selection() != 'exit':
                if selector.is_completed:
                    selector.save_results()
                    selector.get_statistics()
    
    elif choice == '3':
        # 结果查看模式
        output_dir = input("\nEnter output directory (default: ./output): ").strip()
        if not output_dir:
            output_dir = './output'
        
        viewer = ResultViewer(output_dir)
        viewer.interactive_menu()
    
    elif choice == '4':
        print("\n👋 Goodbye!")
        return
    
    else:
        print("✗ Invalid choice")


if __name__ == "__main__":
    main()
