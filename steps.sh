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

# 2.b manually remove the wrong entries in the list.. there are around 10 of them

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
uvx --from topo_map_processor partition --only-disk --from-tiles-dir export/tiles/jica --to-pmtiles-prefix export/pmtiles/Nepal_jica --attribution-file attribution.txt --name "Nepal_jica" --description "Nepal 1:25000 Topo maps from Survey Department in collabartion with JICA"
uvx --from topo_map_processor partition --only-disk --from-tiles-dir export/tiles/main --to-pmtiles-prefix export/pmtiles/Nepal_main --attribution-file attribution.txt --name "Nepal_main" --description "Nepal 1:25000 and 1:50000 Topo maps from Survey Department"

# 9. create the bounds geojson files
uvx --from topo_map_processor collect-bounds --bounds-dir export/bounds/jica --output-file export/bounds_jica.geojson
uvx --from topo_map_processor collect-bounds --bounds-dir export/bounds/main --output-file export/bounds_main.geojson

# 10. create the displayable index files
ogr2ogr -f GeoJSON -s_srs '+proj=tmerc +lat_0=0 +lon_0=84 +k=0.9999 +x_0=500000 +y_0=0 +units=m +ellps=evrst30 +towgs84=293.17,726.18,245.36,0,0,0,0 +no_defs' -t_srs EPSG:4326 data/index_jica_wgs84.geojson data/index_jica.geojsonl
ogr2ogr -f GeoJSON -s_srs EPSG:6207 -t_srs EPSG:4326 data/index_main_wgs84.geojson data/index_main.geojsonl

# 11. upload assets to github
gh release upload maze data/index_jica_wgs84.geojson
gh release upload maze data/index_main_wgs84.geojson
gh release upload maze export/pmtiles/Nepal_jica.pmtiles
gh release upload maze export/pmtiles/Nepal_main.*

gh release upload survey-georef export/gtiffs/main/*.tif
gh release upload jica-georef   export/gtiffs/jica/*.tif

cp export/bounds_main.geojson bounds.geojson
gh release upload survey-georef bounds.geojson
rm bounds.geojson

cp export/bounds_jica.geojson bounds.geojson
gh release upload jica-georef bounds.geojson
rm bounds.geojson

gh release upload survey-orig data/raw/main/*.jpg
gh release upload jica-orig data/raw/jica/*.jpg

# 12. update listing
uvx --from topo_map_processor generate-lists survey-orig .jpg
uvx --from topo_map_processor generate-lists jica-orig .jpg
uvx --from topo_map_processor generate-lists survey-georef .tif
uvx --from topo_map_processor generate-lists jica-georef .tif

# 13. copy the listing files over to pmtiles release to maintain list of files which have been tiled
gh release download survey-georef -p listing_files.csv
mv listing_files.csv listing_files_main.csv
gh release upload maze listing_files_main.csv
rm listing_files_main.csv

gh release download jica-georef -p listing_files.csv
mv listing_files.csv listing_files_jica.csv
gh release upload maze listing_files_jica.csv
rm listing_files_jica.csv
