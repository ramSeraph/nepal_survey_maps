#!/bin/bash

set -e

# 1. download the kml files for jica and the survey maps from https://www.maze.com.np/Maps/Topo-Nepal
mkdir -p data
wget -O data/main.kmz https://www.google.com/maps/d/kml?mid=1mh50CxlPqd7f9OrW_aQ3l0SFA4Bl6e0&resourcekey&forcekml=1
wget -O data/jica.kmz https://www.google.com/maps/d/kml?mid=19bK-IQf12gqMhNOiolRx3kQIIgv__LA&resourcekey&forcekml=1

# 2. Convert the kml files to geojsonl
mkdir -p lists/
ogr2ogr -f "GeoJSONSeq" lists/list_jica.geojsonl data/jica.kmz 
ogr2ogr -f "GeoJSONSeq" lists/list_main.geojsonl data/main.kmz 

# 3. download missing files from pahar
wget -O data/raw/jica/099-12.jpg https://pahar.in/pahar/Maps--Primary/Nepal%20Maps/Nepal%20Topo%20Maps/099-12%20Jhirubas.jpg

wget -O data/raw/main/2687_12B_D.jpg    https://pahar.in/pahar/Maps--Primary/Nepal%20Maps/Nepal%20Topo%20Maps/2687%2012B%20and%20D%20Pathariya.jpg
wget -O data/raw/main/2782_01A_01C.jpg  https://pahar.in/pahar/Maps--Primary/Nepal%20Maps/Nepal%20Topo%20Maps/2782%2001A%20and%2001C%20Bhaisahi%20Naka.jpg
wget -O data/raw/main/2787_16D.jpg      https://pahar.in/pahar/Maps--Primary/Nepal%20Maps/Nepal%20Topo%20Maps/2787%2016D%20Deurali.jpg
wget -O data/raw/main/2883_13B.jpg      https://pahar.in/pahar/Maps--Primary/Nepal%20Maps/Nepal%20Topo%20Maps/2883%2013B%20Tari.jpg

# 4. create the index files for georeferencing at data/
uv run create_main_index.py
uv run create_jica_index.py

# 5. download the files using the lists
uv run download.py

# 6. georeference the images
uv run parse_jica.py
uv run parse_main.py

# 7. create the tiles
GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor tile --tiffs-dir export/gtiffs/jica --tiles-dir export/tiles/jica --max-zoom 15
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor tile --tiffs-dir export/gtiffs/main --tiles-dir export/tiles/main --max-zoom 15

# 8. create the partitioned pmtiles
uvx --from topo_map_processor partition --only-disk --from-tiles-dir export/tiles/jica --to-pmtiles-prefix export/pmtiles/Nepal_jica --max-zoom 15 --attribution-file attribution.txt --name "Nepal_jica" --description "Nepal 1:25000 Topo maps from Survey Department in collabartion with JICA"
uvx --from topo_map_processor partition --only-disk --from-tiles-dir export/tiles/main --to-pmtiles-prefix export/pmtiles/Nepal_main --max-zoom 15 --attribution-file attribution.txt --name "Nepal_main" --description "Nepal 1:25000 and 1:50000 Topo maps from Survey Department"

# 9. create the bounds geojson files
uvx --from topo_map_processor collect-bounds export/bounds/jica export/bounds_jica.geojson
uvx --from topo_map_processor collect-bounds export/bounds/main export/bounds_main.geojson

# 10. create the displayable index files
ogr2ogr -f GeoJSON -t_srs EPSG:4326 data/index_jica_wgs84.geojson data/index_jica.geojsonl
ogr2ogr -f GeoJSON -t_srs EPSG:4326 data/index_main_wgs84.geojson data/index_main.geojsonl

# 11. upload assets to github
gh release upload maze data/index_jica_wgs84.geojson
gh release upload maze data/index_main_wgs84.geojson
gh release upload maze export/pmtiles/Nepal_jica.pmtiles
gh release upload maze export/pmtiles/Nepal_main.*

gh release upload survey-georef export/gtiffs/main/*.tif
gh release upload jica-georef   export/gtiffs/jica/*.tif
gh release upload survey-georef export/bounds_main.geojson#bounds.geojson
gh release upload jica-georef   export/bounds_jica.geojson#bounds.geojson

gh release upload survey-orig data/raw/main/*.jpg
gh release upload jica-orig data/raw/jica/*.jpg

# 12. update listing
./generate_lists.sh survey-orig .jpg
./generate_lists.sh jica-orig .jpg
./generate_lists.sh survey-georef .tif
./generate_lists.sh jica-georef .tif
