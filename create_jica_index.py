from pathlib import Path
import json

index = {
    '091': [3_50_000, 31_50_000],
    '092': [4_00_000, 31_50_000],
    '093': [4_50_000, 31_50_000],
    '097': [3_50_000, 31_00_000],
    '098': [4_00_000, 31_00_000],
    '099': [4_50_000, 31_00_000],
    '100': [5_00_000, 31_00_000],
    '103': [3_50_000, 30_50_000],
    '104': [4_00_000, 30_50_000],
    '105': [4_50_000, 30_50_000],
    '106': [5_00_000, 30_50_000],
}

items = []
for p in Path('data/raw/jica/').glob('*.jpg'):
    name = p.name.replace('.jpg', '')
    #print(name)
    parts = name.split('-')
    main = parts[0]
    sub  = int(parts[1]) - 1

    sub_r = sub // 4
    sub_c = sub % 4

    m_tl = index[main]
    m_tl_e = m_tl[0]
    m_tl_n = m_tl[1]

    tl_e = m_tl_e + (sub_c * 12_500)
    tl_n = m_tl_n - (sub_r * 12_500)

    poly = [[tl_e, tl_n], 
            [tl_e, tl_n - 12_500],
            [tl_e + 12_500, tl_n - 12_500],
            [tl_e + 12_500, tl_n],
            [tl_e, tl_n]]


    item = { 'type': 'Feature', 'properties': { 'Name': name }, 'geometry': { 'type': 'Polygon', 'coordinates': [poly] } }
    items.append(item)

with open('data/index_jica.geojsonl', 'w') as f:
    for item in items:
        f.write(json.dumps(item))
        f.write('\n')

