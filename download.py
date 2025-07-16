# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
# ]
# ///

import requests
import json

from pathlib import Path

data_dir = Path('data')
list_dir = Path('lists')

def get_jica_url_map():
    url_map = {}
    main_list_file = list_dir / 'list_jica.geojsonl'
    with open(main_list_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            props = item['properties']
            name = props['Name']
            link = props.get('description', None)
            if link is None:
                print(f'missing link for {name}')
                continue
            lparts = link.split('/')
            link = f'https://drive.google.com/uc?export=download&id={lparts[5]}'
            parts = name.split(' ')
            key = parts[0]
            url_map[key] = link

    return url_map

def get_main_url_map():
    url_map = {}
    main_list_file = list_dir / 'list_main.geojsonl'
    with open(main_list_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            props = item['properties']
            name = props['Name']
            link = props.get('description', None)
            if link is None or link == 'refer JICA topo map':
                continue
            lparts = link.split('/')
            link = f'https://drive.google.com/uc?export=download&id={lparts[5]}'
            parts = name.split(' ')
            key = '_'.join(parts[:2])
            key = key.replace('+', '_')
            url_map[key] = link

    return url_map

def download_sheets(url_map, typ):
    raw_dir = data_dir / 'raw' / typ
    raw_dir.mkdir(parents=True, exist_ok=True)

    for k, url in url_map.items():
        print(f'downloading {k}')
        resp = requests.get(url)
        if not resp.ok:
            raise Exception(f'Unable to download {url}')

        file = raw_dir / f'{k}.jpg'
        file.write_bytes(resp.content)

def main():
    url_map = get_main_url_map()
    download_sheets(url_map, 'main')
    jica_url_map = get_jica_url_map()
    download_sheets(jica_url_map, 'jica')

if __name__ == '__main__':
    main()
