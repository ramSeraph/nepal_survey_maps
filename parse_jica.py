# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "topo-map-processor",
# ]
#
# ///


import os
import json
from pathlib import Path

from topo_map_processor import TopoMapProcessor, LineRemovalParams

index_map_jica = None
def get_index_map_jica():
    global index_map_jica
    if index_map_jica is not None:
        return index_map_jica

    index_map_jica = {}
    with open('data/index_jica.geojsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            props = item['properties']
            geom  = item['geometry']
            k = props['Name']
            poly = geom['coordinates'][0]
            index_map_jica[k] = poly

    return index_map_jica

class JICAProcessor(TopoMapProcessor):

    def __init__(self, filepath, extra, index_map):
        super().__init__(filepath, extra, index_map)
        self.remove_corner_edges_ratio = extra.get('remove_corner_edges_ratio', 0.5)
        self.remove_corner_text = extra.get('remove_corner_text', True)
        self.corner_contour_color = extra.get('corner_contour_color', 'not_white')
        self.corner_erode = extra.get('corner_erode', -1)

        self.max_corner_contour_area_ratio = extra.get('max_corner_contour_area_ratio', 0.8)
        self.min_corner_contour_area_ratio = extra.get('min_corner_contour_area_ratio', 0.02)

        self.corner_max_dist_ratio = extra.get('corner_max_dist_ratio', 0.2)
        self.max_corner_angle_diff = extra.get('max_corner_angle_diff', 5)
        self.max_corner_angle_diff_cutoff = extra.get('max_corner_angle_diff_cutoff', 10)

        self.find_line_iter = extra.get('find_line_iter', 0)
        self.find_line_scale = extra.get('find_line_scale', 16)
        self.line_color = extra.get('line_color', None)
        self.line_color_choices = extra.get('line_color_choices', [['black'], ['black', 'greyish']])

        self.remove_line_buf_ratio = extra.get('remove_line_buf_ratio', 3.0 / 6500.0)
        self.remove_line_blur_buf_ratio = extra.get('remove_line_blur_buf_ratio', 21.0 / 6500.0)
        self.remove_line_blur_kern_ratio = extra.get('remove_line_blur_kern_ratio', 13.0 / 6500.0)
        self.remove_line_blur_repeat = extra.get('remove_line_blur_repeat', 3)
        self.should_remove_grid_lines = extra.get('should_remove_grid_lines', False)

    def get_inter_dir(self):
        return Path('data/inter/jica')

    def get_gtiff_dir(self):
        return Path('export/gtiffs/jica')

    def get_bounds_dir(self):
        return Path('export/bounds/jica')

    def get_crs_proj(self):
        return '+proj=tmerc +lat_0=0 +lon_0=84 +k=0.9999 +x_0=500000 +y_0=0 +units=m +ellps=evrst30 +towgs84=293.17,726.18,245.36,0,0,0,0 +no_defs'

    def get_scale(self):
        return 25000

    def prompt1(self):
        pass

    def get_intersection_point(self, img, direction, anchor_angle):
        if self.line_color is not None:
            line_color_choices = [ self.line_color ]
        else:
            line_color_choices = self.line_color_choices

        ip = None
        for line_color in line_color_choices:
            try:
                ip = self.get_nearest_intersection_point_from_biggest_corner_contour(
                    img, direction, anchor_angle,
                    line_color, self.corner_contour_color,
                    self.remove_corner_edges_ratio, self.corner_erode,
                    self.max_corner_contour_area_ratio, self.min_corner_contour_area_ratio,
                    self.find_line_scale, self.find_line_iter,
                    self.max_corner_angle_diff_cutoff, self.max_corner_angle_diff,
                    self.corner_max_dist_ratio
                )
                return ip
            except Exception as ex:
                print(f'Failed to find intersection point with color {line_color}: {ex}')

        raise Exception(f'Failed to find intersection point with any of the colors: {line_color_choices}')

    def locate_grid_lines(self):
        gcps = self.get_gcps()
        transformer = self.get_transformer_from_gcps(gcps)

        lines = self.locate_grid_lines_using_trasformer(transformer, 1, 1000)
        params = LineRemovalParams(
            self.remove_line_buf_ratio,
            self.remove_line_blur_buf_ratio,
            self.remove_line_blur_kern_ratio,
            self.remove_line_blur_repeat
        )
        return [ (line, params) for line in lines ]

# TODO: the way corner angles are calculated has changed.. so mught not be reproducabile
def process_files():
    
    data_dir = Path('data/raw/jica')
    
    from_list_file = os.environ.get('FROM_LIST', None)
    if from_list_file is not None:
        fnames = Path(from_list_file).read_text().split('\n')
        image_files = [ Path(f'{data_dir}/{f.strip()}') for f in fnames if f.strip() != '']
    else:
        # Find all jpg files
        print(f"Finding jpg files in {data_dir}")
        image_files = list(data_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} jpg files")
    
    
    special_cases_file = Path(__file__).parent / 'special_cases_jica.json'

    special_cases = {}
    if special_cases_file.exists():
        special_cases = json.loads(special_cases_file.read_text())

    total = len(image_files)
    processed_count = 0
    failed_count = 0
    success_count = 0
    # Process each file
    for filepath in image_files:
        print(f'==========  Processed: {processed_count}/{total} Success: {success_count} Failed: {failed_count} processing {filepath.name} ==========')
        extra = special_cases.get(filepath.name, {})
        index_map = get_index_map_jica()
        processor = JICAProcessor(filepath, extra, index_map)

        try:
            processor.process()
            success_count += 1
        except Exception as ex:
            print(f'parsing {filepath} failed with exception: {ex}')
            failed_count += 1
            import traceback
            traceback.print_exc()
            raise
            processor.prompt()
        processed_count += 1

    print(f"Processed {processed_count} images, failed_count {failed_count}, success_count {success_count}")


if __name__ == "__main__":

    process_files()
    
