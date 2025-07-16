#!/bin/bash

release=$1
ext=$2

if [ -z "$release" ] || [ -z "$ext" ]; then
  echo "Usage: $0 <release> <extension>"
  exit 1
fi

owner=$(gh repo view --json owner -q .owner.login)
name=$(gh repo view --json name -q .name)
repo="$owner/$name"

echo "getting file list"
gh release view $release --json assets -q '.assets[] | "\(.size) \(.name)"' -R $repo | grep "${ext}\$" > listing_files.txt
cat listing_files.txt | cut -d" " -f2 | xargs -I {} echo "https://github.com/$repo/releases/download/$release/{}" > url_list.txt

echo "uploading listing"
gh release upload $release listing_files.txt --clobber -R $repo
gh release upload $release url_list.txt --clobber -R $repo

rm listing_files.txt url_list.txt
