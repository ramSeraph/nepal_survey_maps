#!/bin/bash

# this assumes that the raw files and files in the export directory have been cleared

# 1) create a retile file list at from_list.txt with the 'jpg' extensions 
#

# 2) download the raw files
mkdir -p data/raw/main
cat from_list.txt | xargs -I {} gh release download survey-orig -D data/raw/main/ -p {}

# 3) run the parse script with the files
uv run parse_main.py

# 4) upload files
gh release upload survey-georef export/gtiffs/main/*.tif --clobber

# 5) update the bounds.geojson
gh release download survey-georef -p bounds.geojson
mv bounds.geojson export/bounds_main.geojson
uvx --from topo_map_processor collect-bounds --preexisting-file export/bounds_main.geojson --bounds-dir export/bounds/main --output-file bounds.geojson
gh release upload survey-georef bounds.geojson --clobber
rm bounds.geojson
rm export/bounds_main.geojson

# 6) update listing
uvx --from topo_map_processor generate-lists survey-georef .tif

# 7) redo the tiling 
GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)
uvx --with numpy \
    --with pillow \
    --with gdal==$GDAL_VERSION \
    --from topo_map_processor \
    retile-e2e -p maze -g survey-georef -x Nepal_main -l listing_files_main.csv


