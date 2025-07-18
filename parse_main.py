# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "topo-map-processor",
# ]
#
# ///


import os
import re
import json
import traceback
from pathlib import Path

from shapely.geometry import (
    JOIN_STYLE, Point, Polygon
)
from pyproj import Transformer

from topo_map_processor import TopoMapProcessor, LineRemovalParams

index_map_main = None
def get_index_map_main():
    global index_map_main
    if index_map_main is not None:
        return index_map_main

    index_map_main = {}
    with open('data/index_main.geojsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            props = item['properties']
            geom  = item['geometry']
            k = props['Name']
            poly = geom['coordinates'][0]
            index_map_main[k] = poly

    return index_map_main


class MainProcessor(TopoMapProcessor):

    def __init__(self, filepath, extra, index_map):
        super().__init__(filepath, extra, index_map)

        self.band_color = extra.get('band_color', None)
        self.band_color_choices = extra.get('band_color_choices', [['black'], ['black', 'greyish'], ['not_white']])

        # grid line removal related
        # too many leftover lines, so giving up
        self.should_remove_grid_lines = extra.get('should_remove_grid_lines', False)

        self.remove_meter_line_buf_ratio = extra.get('remove_meter_line_buf_ratio', 3.0 / 6500.0)
        self.remove_meter_line_blur_buf_ratio = extra.get('remove_meter_line_blur_buf_ratio', 20.0 / 6500.0)
        self.remove_meter_line_blur_kern_ratio = extra.get('remove_meter_line_blur_kern_ratio', 13.0 / 6500.0)
        self.remove_meter_line_blur_repeat = extra.get('remove_meter_line_blur_repeat', 6)

        self.remove_degree_line_buf_ratio = extra.get('remove_degree_line_buf_ratio', 3.0 / 6500.0)
        self.remove_degree_line_blur_buf_ratio = extra.get('remove_degree_line_blur_buf_ratio', 20.0 / 6500.0)
        self.remove_degree_line_blur_kern_ratio = extra.get('remove_degree_line_blur_kern_ratio', 13.0 / 6500.0)
        self.remove_degree_line_blur_repeat = extra.get('remove_degree_line_blur_repeat', 6)

        self.grid_bounds_check_buffer_ratio = extra.get('grid_bounds_check_buffer_ratio', 40.0 / 7000.0)
        self.grid_line_correction_context_ratio_1 = extra.get('grid_line_correction_context_ratio_1', 15.0 / 7000.0 )
        self.grid_line_correction_context_ratio_2 = extra.get('grid_line_correction_context_ratio_2', 10.0 / 7000.0 )
        self.grid_line_color_degree_choices = extra.get('grid_line_color_degree', ['black'])
        self.grid_line_color_meter_choices = extra.get('grid_line_color_meter', ['sky_blue'])
        self.grid_find_line_iter = extra.get('grid_find_line_iter', 0)
        self.grid_find_line_scale = extra.get('grid_find_line_scale', 2)



        self.remove_corner_edges_ratio = extra.get('remove_corner_edges_ratio', 0.3)
        self.remove_text_for_corner_contour = extra.get('remove_text_for_corner_contour', True)
        self.text_removal_engine = extra.get('text_removal_engine', 'easyocr')
        # self.text_removal_iterations = extra.get('text_removal_iterations', -2)

        self.corner_contour_color = extra.get('corner_contour_color', 'not_white')
        self.corner_erode = extra.get('corner_erode', -1)
        #self.picked_corner_max_dist_from_contour_ratio = extra.get('picked_corner_max_dist_from_contour_ratio', 3.0 / 400.0)
        self.picked_corner_max_dist_from_contour_ratio = extra.get('picked_corner_max_dist_from_contour_ratio', 4.0 / 400.0)
        self.min_corner_dist_ratio = extra.get('min_corner_dist_ratio', 0.2)
        self.max_corner_dist_ratio = extra.get('max_corner_dist_ratio', 0.75)

        self.max_corner_contour_area_ratio = extra.get('max_corner_contour_area_ratio', 0.8)
        self.min_corner_contour_area_ratio = extra.get('min_corner_contour_area_ratio', 0.065)

        self.pixel_adjustment = extra.get('pixel_adjustment', 3)
        self.max_corner_angle_diff = extra.get('max_corner_angle_diff', 4)
        self.max_contour_corner_angle_diff = extra.get('max_contour_corner_angle_diff', 5)
        self.max_corner_angle_diff_cutoff = extra.get('max_corner_angle_diff_cutoff', 20)

        self.find_line_iter = extra.get('find_line_iter', 2)
        self.find_line_scale = extra.get('find_line_scale', 8)

        self.line_color = extra.get('line_color', None)
        self.line_color_choices = extra.get('line_color_choices', [['black'], ['black', 'greyish'], ['not_white']])
        #self.color_map['sky_blue'] = ((80, 50, 0), (140, 255, 255))
        self.color_map['sky_blue'] = ((80, 30, 0), (140, 255, 255))
        # H: 101.5, S: 102, V: 173
        # H: 202, S:31, V: 76


    def get_inter_dir(self):
        return Path('data/inter/main')

    def get_gtiff_dir(self):
        return Path('export/gtiffs/main')

    def get_bounds_dir(self):
        return Path('export/bounds/main')

    def get_crs_proj_for_meter_lines(self):
        ibox = self.get_sheet_ibox()

        p1 = ibox[0]
        p2 = ibox[2]

        long_mid = (p1[0] + p2[0]) / 2

        if long_mid > 85.5:
           long_base = 87
        elif long_mid < 82.5:
           long_base = 81
        else:
           long_base = 84

        return f'+proj=tmerc +lat_0=0 +lon_0={long_base} +k=0.9999 +x_0=500000 +y_0=0 +units=m +ellps=evrst30 +towgs84=293.17,726.18,245.36,0,0,0,0 +no_defs'


    def get_crs_proj(self):
        return 'EPSG:6207'

    def save_points(self, corrections, save_to_file_name_prefix):
        color_map = {
            'invalid': (0, 0, 0),
            'no_choices': (255, 0, 0),
            'too_many_choices': (0, 255, 255),
            'unchanged': (0, 255, 0),
            'corrected': (0, 0, 255)
        }
        corrected_points_and_colors = [ (c['point'], color_map[c['type']]) for p, c in corrections.items() ]
        #print(f'corrected_points_and_colors: {len(corrected_points_and_colors)}')
        self.save_with_points(corrected_points_and_colors,
                              self.get_full_img(), 
                              self.get_workdir() / f'{save_to_file_name_prefix}.jpg',
                              radius=2)


    def get_corrections(self, lines, color_choices,
                        context_dim, bounds_check_buffer,
                        save_to_file_name_prefix):

        points = set()
        for i, line in enumerate(lines):
            points.add(line[0])
            points.add(line[1])

        corrections = self.get_grid_line_corrections(points, bounds_check_buffer,
                                                     color_choices, context_dim,
                                                     self.grid_find_line_scale, self.grid_find_line_iter)

        corrections_by_type = {}
        for point, correction in corrections.items():
            if correction['type'] not in corrections_by_type:
                corrections_by_type[correction['type']] = {}
            corrections_by_type[correction['type']][point] = correction

        for correction_type, elems in corrections_by_type.items():
            print(f'correction type: {correction_type}, count: {len(elems)}')

        self.save_points(corrections, save_to_file_name_prefix)
        return corrections_by_type


    def locate_grid_lines_internal(self, gcps, bounds_check_buffer,
                                   context_dim_1, context_dim_2, 
                                   factor, interval,
                                   color_choices, name):
        xys_in_gcps = set([tuple(gcp[1]) for gcp in gcps])
        pixel_transformer = self.get_transformer_from_gcps(gcps)
        lines, lines_xy = self.locate_grid_lines_using_trasformer(pixel_transformer, factor, interval, bounds_check_buffer)

        to_xy = {}
        for i, line in enumerate(lines):
            line_xy = lines_xy[i]
            to_xy[line[0]] = line_xy[0]
            to_xy[line[1]] = line_xy[1]

        corrections_by_type = self.get_corrections(lines, color_choices,
                                                   context_dim_1, bounds_check_buffer,
                                                   f'grid_points_{name}')
        to_use = {}
        to_use.update(corrections_by_type.get('corrected', {}))
        to_use.update(corrections_by_type.get('unchanged', {}))
        for p, correction in to_use.items():
            xy = to_xy[p]
            if tuple(xy) in xys_in_gcps:
                continue
            gcps.append([correction['point'], xy])

        pixel_transformer = self.get_transformer_from_gcps(gcps)
        lines, lines_xy = self.locate_grid_lines_using_trasformer(pixel_transformer, factor, interval, bounds_check_buffer)
        corrections_by_type = self.get_corrections(lines, color_choices,
                                                   context_dim_2, bounds_check_buffer,
                                                   f'grid_points_{name}_1')
        corrections = {}
        for _, items in corrections_by_type.items():
            corrections.update(items)

        lines = [ (corrections[line[0]]['point'], corrections[line[1]]['point']) for line in lines ]

        return lines


    def locate_grid_lines(self):
        full_img = self.get_full_img()
        h,w = full_img.shape[:2]
        ratio_1 = self.grid_line_correction_context_ratio_1
        ratio_2 = self.grid_line_correction_context_ratio_2
        context_dim_1 = int(h * ratio_1), int(w * ratio_1)
        context_dim_2 = int(h * ratio_2), int(w * ratio_2)

        bounds_check_buffer = int(self.grid_bounds_check_buffer_ratio * h)

        meter_crs = self.get_crs_proj_for_meter_lines()
        geo_crs = self.get_crs_proj()
        transformer = Transformer.from_crs(geo_crs, meter_crs, always_xy=True)

        gcps = self.get_gcps()

        meter_gcps = []
        for gcp in gcps:
            corner = gcp[0]
            idx = gcp[1]
            meter_idx = transformer.transform(idx[0], idx[1])
            meter_gcp = [corner, meter_idx]
            meter_gcps.append(meter_gcp)

        degree_factor = 12 if self.get_scale() == 50000 else 24
        print(f'Using degree factor: {degree_factor} for scale {self.get_scale()}')

        degree_lines = self.locate_grid_lines_internal(gcps, bounds_check_buffer, context_dim_1, context_dim_2,
                                                       degree_factor, 1, self.grid_line_color_degree_choices,
                                                       'degree')

        meter_lines = self.locate_grid_lines_internal(meter_gcps, bounds_check_buffer, context_dim_1, context_dim_2,
                                                      1, 1000, self.grid_line_color_meter_choices,
                                                      'meter')

        meter_params = LineRemovalParams(
            self.remove_meter_line_buf_ratio,
            self.remove_meter_line_blur_buf_ratio,
            self.remove_meter_line_blur_kern_ratio,
            self.remove_meter_line_blur_repeat
        )
        meter_lines = [ (line, meter_params) for line in meter_lines ]

        degree_params = LineRemovalParams(
            self.remove_degree_line_buf_ratio,
            self.remove_degree_line_blur_buf_ratio,
            self.remove_degree_line_blur_kern_ratio,
            self.remove_degree_line_blur_repeat
        )

        degree_lines = [ (line, degree_params) for line in degree_lines ]

        return meter_lines + degree_lines


    def get_scale(self):
        if re.search(r'[A-Z]', self.get_id()):
            return 25000
        return 50000

    def get_intersection_point(self, img, direction, anchor_angle):

        ip = None

        try:
            remove_text = self.remove_text_for_corner_contour
            ip = self.get_biggest_contour_corner(img, direction, anchor_angle,
                                                 self.corner_contour_color, remove_text,
                                                 self.remove_corner_edges_ratio,
                                                 self.corner_erode, self.max_corner_contour_area_ratio,
                                                 self.min_corner_contour_area_ratio, self.min_corner_dist_ratio,
                                                 self.max_contour_corner_angle_diff, self.pixel_adjustment,
                                                 self.picked_corner_max_dist_from_contour_ratio)
            return ip
        except Exception as ex:
            print(f'Failed to find corner contour: {ex}')
            traceback.print_exc()


        if self.line_color is not None:
            line_color_choices = [self.line_color]
        else:
            line_color_choices = self.line_color_choices

        for line_color in line_color_choices:
            try:
                min_expected_points = 1
                expect_band_count = 1
                remove_text = False
                ip = self.get_nearest_intersection_point(img, direction, anchor_angle,
                                                         line_color, remove_text, expect_band_count,
                                                         self.find_line_scale, self.find_line_iter,
                                                         self.max_corner_dist_ratio, self.min_corner_dist_ratio,
                                                         min_expected_points, self.max_corner_angle_diff,
                                                         self.max_corner_angle_diff_cutoff)
                return ip
            except Exception as ex:
                print(f'Failed to nearest intersection: {ex}')

        if ip is None:
            raise Exception('Failed to find intersection point')

        return ip

    def prompt1(self):
        pass


def process_files():
    
    data_dir = Path('data/raw/main')
    
    from_list_file = os.environ.get('FROM_LIST', None)
    if from_list_file is not None:
        fnames = Path(from_list_file).read_text().split('\n')
        image_files = [ Path(f'{data_dir}/{f.strip()}') for f in fnames if f.strip() != '']
    else:
        # Find all jpg files
        print(f"Finding jpg files in {data_dir}")
        image_files = list(data_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} jpg files")
    
    
    special_cases_file = Path(__file__).parent / 'special_cases_main.json'

    special_cases = {}
    if special_cases_file.exists():
        special_cases = json.loads(special_cases_file.read_text())

    index_map = get_index_map_main()

    total = len(image_files)
    processed_count = 0
    failed_count = 0
    success_count = 0
    # Process each file
    for filepath in image_files:
        print(f'==========  Processed: {processed_count}/{total} Success: {success_count} Failed: {failed_count} processing {filepath.name} ==========')
        extra = special_cases.get(filepath.name, {})
        id = filepath.name.replace('.jpg', '')
        index_box = index_map[id]
        processor = MainProcessor(filepath, extra, index_box)

        try:
            #if filepath.name in [
            #    '2784_10A_C.jpg',
            #    '2984_13.jpg',
            #]:
            #    continue
            processor.process()
            success_count += 1
        except Exception as ex:
            print(f'parsing {filepath} failed with exception: {ex}')
            failed_count += 1
            traceback.print_exc()
            raise
            processor.prompt()
        processed_count += 1

    print(f"Processed {processed_count} images, failed_count {failed_count}, success_count {success_count}")


if __name__ == "__main__":
    process_files()
    
