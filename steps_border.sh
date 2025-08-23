# download pahar page
#
uv run download_border_sheets.py

uv run parse_border.py

GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo-map-processor tile  --tiffs-dir export/gtiffs/border --tiles-dir export/tiles/border --max-zoom 14 --attribution-file attribution_border.txt --name "Nepal_border" --description "Nepal China Boundary 1979"

uvx --from topo-map-processor collect-bounds --bounds-dir export/bounds/border --output-file export/bounds_border.geojson

uvx --from pmtiles-mosaic partition --from-source export/tiles/border --to-pmtiles export/pmtiles/Nepal_border.pmtiles

gh release upload border-orig data/raw/border/sheet_*
uvx --from gh-release-tools generate-lists -r 'border-orig' -e '.jpg'

gh release upload border-georef export/gtiffs/border/sheet_*
uvx --from gh-release-tools generate-lists -r 'border-georef' -e '.tif'

cp export/bounds_border.geojson bounds.geojson
gh release upload border-georef bounds.geojson
rm bounds.geojson

gh release upload border export/pmtiles/Nepal_border.pmtiles

gh release download border-georef -p listing_files.csv
gh release upload border listing_files.csv
rm listing_files.csv
