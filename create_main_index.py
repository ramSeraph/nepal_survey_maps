import json


def write_index():
    out = []
    with open('lists/list_main.geojsonl', 'r') as f:
        for line in f:
            item = json.loads(line)

            props = item['properties']

            name = props['Name']
            parts = name.split(' ')
            name = '_'.join(parts[:2])
            name = name.replace('+', '_')

            item['properties'] = { 'Name': name }

            geom = item['geometry']
            coords = geom['coordinates']
            new_poly = []
            for p in coords[0]:
                np = p[:2]
                new_poly.append(np)

            geom['coordinates'] = [ new_poly ]

            out.append(item)

    with open('data/index_main.geojsonl', 'w') as f:
        for item in out:
            f.write(json.dumps(item))
            f.write('\n')


if __name__ == '__main__':
    write_index()
