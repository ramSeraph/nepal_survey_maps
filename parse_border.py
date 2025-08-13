# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "topo-map-processor[parse]",
#     "easyocr",
# ]
#
# [tool.uv.sources]
# topo_map_processor = { path = "../topo_map_processor/", editable = true }
#
# ///


import os
import json
from pathlib import Path

import easyocr
import cv2

from topo_map_processor.processor import TopoMapProcessor

easy_ocr_reader = None

class BorderProcessor(TopoMapProcessor):

    def __init__(self, filepath, extra, index_map):
        super().__init__(filepath, extra, index_map)
        self.remove_corner_edges_ratio = extra.get('remove_corner_edges_ratio', 0.25)
        self.remove_corner_text = extra.get('remove_corner_text', True)
        self.text_removal_engine = extra.get('text_removal_engine', 'easyocr')
        self.corner_contour_color = extra.get('corner_contour_color', 'black')
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
        #self.line_color_choices = extra.get('line_color_choices', ['black'])
        self.ext_thresh_ratio = extra.get('ext_thresh_ratio', 20.0 / 18000.0)
        self.cwidth = extra.get('cwidth', 1)

        self.remove_line_buf_ratio = extra.get('remove_line_buf_ratio', 3.0 / 6500.0)
        self.remove_line_blur_buf_ratio = extra.get('remove_line_blur_buf_ratio', 21.0 / 6500.0)
        self.remove_line_blur_kern_ratio = extra.get('remove_line_blur_kern_ratio', 13.0 / 6500.0)
        self.remove_line_blur_repeat = extra.get('remove_line_blur_repeat', 3)
        self.should_remove_grid_lines = extra.get('should_remove_grid_lines', False)

        self.index_box_unprocessed = []


    def get_inter_dir(self):
        return Path('data/inter/border')

    def get_gtiff_dir(self):
        return Path('export/gtiffs/border')

    def get_bounds_dir(self):
        return Path('export/bounds/border')

    def get_crs_proj(self):
        return 'EPSG:4610'

    def get_scale(self):
        return 50000

    def prompt1(self):
        pass

    def get_coordinates(self, img, ip, direction, anchor_angle):
        global easy_ocr_reader
 
        if easy_ocr_reader is None:
            easy_ocr_reader = easyocr.Reader(['en'], gpu=False)
    
        print("Detecting text...")
        # erode the image to remove noise

        #size = 1
        #el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        #img = cv2.erode(img, el)

        # split image into 4 around ip
        h, w = img.shape[:2]
        x, y = ip
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= w or y >= h:
            raise Exception(f"Intersection point {ip} is out of bounds for image size {w}x{h}")
        img_left_above = img[:y, :x]
        img_left_below = img[y:, :x]
        img_right_above = img[:y, x:]
        img_right_below = img[y:, x:]

        left_above_results = easy_ocr_reader.readtext(img_left_above)
        left_below_results = easy_ocr_reader.readtext(img_left_below)
        right_above_results = easy_ocr_reader.readtext(img_right_above)
        right_below_results = easy_ocr_reader.readtext(img_right_below)

        all_results = []
        for results, position in zip(
            [left_above_results, left_below_results, right_above_results, right_below_results],
            ['left_above', 'left_below', 'right_above', 'right_below']):
            print(f"Found {len(results)} text regions in {position} image")
            new_results = []
            for (bbox, text, confidence) in results:
                center = ((bbox[0][0] + bbox[2][0]) / 2, (bbox[0][1] + bbox[2][1]) / 2)
                new_results.append((bbox, text, confidence, position, center))
                print(f"Position: {position}, Text: '{text}' BBox: '{bbox}' (Confidence: {confidence:.2f})")
            all_results.append(new_results)

        left_above_results, left_below_results, right_above_results, right_below_results = all_results

        print(direction, ip)

        def to_num(txt):
            # replace all non digits and non-decimal points with empty string
            only_digits = ''.join(c for c in txt if c.isdigit())
            only_digits = only_digits.replace('o', '0')

            if len(only_digits) == 4:
                return int(only_digits[:2]) + int(only_digits[2:])/60

            if len(only_digits) == 3:
                only_digits = only_digits[:-1]

            return int(only_digits)

        DUMMY_TXT = '200'
        DUMMY_BBOX = [[0, 0], [0, 0], [0, 0], [0, 0]]
        DUMMY_CENTER = (x/2, y/2)
        DUMMY_RESULT = (DUMMY_BBOX, DUMMY_TXT, 1.0, 'dummy', DUMMY_CENTER)
        if direction == (1, 1):
            if len(left_above_results) != 2:
                left_above_results = [DUMMY_RESULT] * 2
                #raise Exception(f"Expected 2 left above text region, but found {len(left_above_results)}. Please check the image.")
            left_above_results.sort(key=lambda i: i[4][0])  # sort by x coordinate

            if len(left_below_results) != 1:
                left_below_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 left below text region, but found {len(left_below_results)}. Please check the image.")

            if len(right_above_results) != 1:
                right_below_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 right above text regions, but found {len(right_above_results)}. Please check the image.")


            lon_min = to_num(right_above_results[0][1])
            lat_min = to_num(left_below_results[0][1])
            lon_deg = to_num(left_above_results[-1][1])
            lat_deg = to_num(left_above_results[0][1])

            print(f"Coordinates: lon={lon_deg}.{lon_min}, lat={lat_deg}.{lat_min}")
            return [
                (lon_deg, lon_min),
                (lat_deg, lat_min),
            ]
        elif direction == (1, -1):
            if len(left_above_results) != 1:
                left_above_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 left above text region, but found {len(left_above_results)}. Please check the image.")

            if len(left_below_results) != 2:
                left_below_results = [DUMMY_RESULT] * 2
                #raise Exception(f"Expected 2 left below text region, but found {len(left_below_results)}. Please check the image.")
            left_below_results.sort(key=lambda i: i[4][0])

            if len(right_below_results) != 1:
                right_below_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 right below text regions, but found {len(right_below_results)}. Please check the image.")

            lon_min = to_num(right_below_results[0][1])
            lat_min = to_num(left_below_results[0][1])
            lon_deg = to_num(left_below_results[-1][1])
            lat_deg = to_num(left_above_results[0][1])

 
            print(f"Coordinates: lon={lon_deg}.{lon_min}, lat={lat_deg}.{lat_min}")
            return [
                (lon_deg, lon_min),
                (lat_deg, lat_min),
            ]
        elif direction == (-1, -1):
            if len(right_above_results) != 1:
                right_above_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 right above text region, but found {len(right_above_results)}. Please check the image.")

            if len(left_below_results) != 1:
                left_below_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 left below text region, but found {len(left_below_results)}. Please check the image.")

            if len(right_below_results) != 2:
                right_below_results = [DUMMY_RESULT] * 2
                #raise Exception(f"Expected 2 right below text regions, but found {len(right_below_results)}. Please check the image.")
            right_below_results.sort(key=lambda i: i[4][0])

            lon_min = to_num(right_below_results[0][1])
            lon_deg = to_num(left_below_results[0][1])
            lat_deg = to_num(right_above_results[0][1])
            lat_min = to_num(right_below_results[-1][1])

            print(f"Coordinates: lon={lon_deg}.{lon_min}, lat={lat_deg}.{lat_min}")
            return [
                (lon_deg, lon_min),
                (lat_deg, lat_min),
            ]
        elif direction == (-1, 1):
            if len(right_above_results) != 2:
                right_above_results = [DUMMY_RESULT] * 2
                #raise Exception(f"Expected 2 right above text region, but found {len(right_above_results)}. Please check the image.")
            right_above_results.sort(key=lambda i: i[4][0])  # sort by x coordinate

            if len(left_above_results) != 1:
                left_above_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 left above text region, but found {len(left_below_results)}. Please check the image.")

            if len(right_below_results) != 1:
                right_below_results = [DUMMY_RESULT]
                #raise Exception(f"Expected 1 right below text regions, but found {len(right_below_results)}. Please check the image.")

            lon_min = to_num(right_above_results[0][1])
            lat_min = to_num(right_below_results[0][1])
            lon_deg = to_num(left_above_results[0][1])
            lat_deg = to_num(right_above_results[-1][1])

            print(f"Coordinates: lon={lon_deg}.{lon_min}, lat={lat_deg}.{lat_min}")
            return [ 
                (lon_deg, lon_min),
                (lat_deg, lat_min),
            ]

    def precess_index_box(self):
        if len(self.index_box_unprocessed) != 4:
            raise Exception(f"Expected 4 index box coordinates, but found {len(self.index_box_unprocessed)}. Please check the image.")

        print('Processing index box coordinates:', self.index_box_unprocessed)

        box = []
        for i in range(4):
            lon_val, lat_val = self.index_box_unprocessed[i]
            lon_deg = lon_val[0]
            lon_min = lon_val[1]
            if lon_deg < 80 or lon_deg > 89 or lon_min < 0 or lon_min >= 60:
                lon = None
            else:
                lon = lon_deg + lon_min / 60.0

            lat_deg = lat_val[0]
            lat_min = lat_val[1]
            if lat_deg < 27 or lat_deg > 31 or lat_min < 0 or lat_min >= 60:
                lat = None
            else:
                lat = lat_deg + lat_min / 60.0
            box.append((lon, lat))

        print('Box coordinates before processing:', box)

        # auto correct none values if alternate values are present
        for i in range(4):
            if i % 2 == 0:
                if box[i][0] is None and box[(i + 1) % 4][0] is not None:
                    box[i] = (box[(i + 1) % 4][0], box[i][1])
                if box[i][1] is None and box[(i + 3) % 4][1] is not None:
                    box[i] = (box[i][0], box[(i + 3) % 4][1])
            else:
                if box[i][1] is None and box[(i + 1) % 4][1] is not None:
                    box[i] = (box[i][0], box[(i + 1) % 4][1])
                if box[i][0] is None and box[(i + 3) % 4][0] is not None:
                    box[i] = (box[(i + 3) % 4][0], box[i][1])

        self.index_box = box + [box[0]]  # close the box
        print('Index box coordinates:', self.index_box)

    def get_intersection_point(self, img, direction, anchor_angle):
        if self.line_color is not None:
            line_color_choices = [ self.line_color ]
        else:
            line_color_choices = self.line_color_choices

        w, h = self.get_full_img().shape[:2]
        ext_thresh = self.ext_thresh_ratio * w
        ip = None
        for line_color in line_color_choices:
            try:
                ip = self.get_4way_intersection_point(img, line_color, self.remove_corner_text,
                                                      self.find_line_scale, self.find_line_iter,
                                                      self.cwidth, ext_thresh,
                                                      direction,
                                                      self.remove_corner_edges_ratio)
                self.index_box_unprocessed.append(self.get_coordinates(img, ip, direction, anchor_angle))
                if len(self.index_box_unprocessed) == 4:
                    self.precess_index_box()
                return ip
            except Exception as ex:
                print(f'Failed to find intersection point with color {line_color}: {ex}')

        raise Exception(f'Failed to find intersection point with any of the colors: {line_color_choices}')


def process_files():
    
    data_dir = Path('data/raw/border')
    
    from_list_file = os.environ.get('FROM_LIST', None)
    if from_list_file is not None:
        fnames = Path(from_list_file).read_text().split('\n')
        image_files = [ Path(f'{data_dir}/{f.strip()}') for f in fnames if f.strip() != '']
    else:
        # Find all jpg files
        print(f"Finding jpg files in {data_dir}")
        image_files = list(data_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} jpg files")
    
    
    special_cases_file = Path(__file__).parent / 'special_cases_border.json'

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
        id = filepath.name.replace('.jpg', '')

        processor = BorderProcessor(filepath, extra, [])

        try:
            processor.process()
            success_count += 1
        except Exception as ex:
            print(f'parsing {filepath} failed with exception: {ex}')
            failed_count += 1
            import traceback
            traceback.print_exc()
            #raise
            processor.prompt()
        processed_count += 1

    print(f"Processed {processed_count} images, failed_count {failed_count}, success_count {success_count}")


if __name__ == "__main__":

    process_files()
    
