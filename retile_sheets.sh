#!/bin/bash

set -e

# Parse command line arguments.
# Usage: ./retile_main.sh -p <pmtiles_release> -g <gtiffs_release> -x <pmtiles_prefix> -l <listing_files_tiled>
while getopts p:g:x:l: flag
do
    case "${flag}" in
        p) pmtiles_release=${OPTARG};;
        g) gtiffs_release=${OPTARG};;
        x) pmtiles_prefix=${OPTARG};;
        l) listing_files_tiled=${OPTARG};;
    esac
done

if [ -z "$pmtiles_release" ] || [ -z "$gtiffs_release" ] || [ -z "$pmtiles_prefix" ] || [ -z "$listing_files_tiled" ]; then
    echo "Usage: $0 -p <pmtiles_release> -g <gtiffs_release> -x <pmtiles_prefix> -l <listing_files_tiled>"
    exit 1
fi




# name of the listing file for tiled PMTiles in the release
sheets_to_pull_list_outfile="sheets_to_pull_list.txt"
tiles_dir="staging/tiles/"
tiffs_dir="staging/gtiffs/"
from_pmtiles_dir="staging/pmtiles"
to_pmtiles_dir="export/pmtiles"
retile_list_file=to_retile.txt

from_pmtiles_prefix="${from_pmtiles_dir}/${pmtiles_prefix}"
to_pmtiles_prefix="${to_pmtiles_dir}/${pmtiles_prefix}"

echo "Downloading the listing files to get the list of sheets to retile"
gh release download $gtiffs_release -p listing_files.txt --clobber
gh release download $pmtiles_release -p $listing_files_tiled --clobber -O listing_files_tiled.txt

comm <(cat listing_files.txt| cut -d" " -f2 | sort) <(cat listing_files_tiled.txt | cut -d" " -f2 | sort) | cut -f1 | grep "^[0-9]" > $retile_list_file

rm listing_files.txt listing_files_tiled.txt

num_sheets=$(cat $retile_list_file | wc -l)
if [ $num_sheets -eq 0 ]; then
    echo "No sheets to retile found. Exiting."
    rm $retile_list_file
    exit 0
fi

echo "Downloading the bounds file"
gh release download $gtiffs_release -p bounds.geojson

mkdir -p $from_pmtiles_dir
echo "Getting original PMTiles files"
gh release download $pmtiles_release -D "$from_pmtiles_dir" -p ${pmtiles_prefix}*

GDAL_VERSION=$(gdalinfo --version | cut -d"," -f1 | cut -d" " -f2)

# get the list of sheets to pull
echo "Getting list of sheets to pull"
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor retile \
          --retile-list-file "$retile_list_file" \
          --bounds-file bounds.geojson \
          --sheets-to-pull-list-outfile $sheets_to_pull_list_outfile \
          --from-pmtiles-prefix $from_pmtiles_prefix \
          --tiles-dir $tiles_dir \
          --tiffs-dir $tiffs_dir

echo "Sheets to pull:"
cat $sheets_to_pull_list_outfile

sheet_patterns=''
for sheet in $(cat $sheets_to_pull_list_outfile); do
    sheet_patterns="${sheet_patterns} -p ${sheet}"
done
echo "Pulling sheets"
gh release download $gtiffs_release -D "$tiffs_dir" $sheet_patterns
rm $sheets_to_pull_list_outfile

echo "Retiling the sheets"
uvx --with numpy --with pillow --with gdal==$GDAL_VERSION --from topo_map_processor retile \
          --retile-list-file "$retile_list_file" \
          --bounds-file bounds.geojson \
          --from-pmtiles-prefix $from_pmtiles_prefix \
          --tiles-dir $tiles_dir \
          --tiffs-dir $tiffs_dir

rm $retile_list_file

echo "Create the new pmtiles files"
uvx --from topo_map_processor partition \
                   --from-tiles-dir $tiles_dir \
                   --from-pmtiles-prefix $from_pmtiles_prefix \
                   --to-pmtiles-prefix $to_pmtiles_prefix


# the new list might not match previous one.. delete and upload, to avoid leftovers
echo "Deleting old PMTiles files from the release"
cd $from_pmtiles_dir
for fname in ${pmtiles_prefix}*; do
  gh release delete-asset $pmtiles_release $fname -y 
done
cd -

echo "Uploading the new PMTiles files"
gh release upload $pmtiles_release ${to_pmtiles_prefix}* --clobber

echo "Creating the listing files for PMTiles from the geotiffs release"
gh release download $gtiffs_release -p listing_files.txt
if [[ $listing_files_tiled != "listing_files.txt" ]]; then
  echo "Renaming listing_files.txt to $listing_files_tiled"
  mv listing_files.txt $listing_files_tiled
fi

gh release upload $pmtiles_release $listing_files_tiled --clobber
rm $listing_files_tiled

# cleanup
echo "Cleaning up staging directories"
echo "Deleting bounds.geojson"
rm bounds.geojson

echo "Deleting $tiles_dir"
rm -rf $tiles_dir

echo "Deleting $tiffs_dir"
rm -rf $tiffs_dir

echo "Deleting ${from_pmtiles_prefix}*"
rm -rf ${from_pmtiles_prefix}*

echo "Deleting ${to_pmtiles_prefix}*"
rm -rf ${to_pmtiles_prefix}*
