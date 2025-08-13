# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bs4",
#     "requests",
# ]
# ///


from pathlib import Path
from bs4 import BeautifulSoup
import requests

def get_border_page(session):
    url = 'https://pahar.in/nepal-tibet-border-set-of-57-maps-1979/'
    out_file = Path('data/border_sheets.html')
    if out_file.exists():
        return out_file.read_text()
    resp = session.get(url)
    resp.raise_for_status()  # Ensure we got a valid response
    out_file.write_text(resp.text)
    return resp.text

if __name__ == "__main__":
    session = requests.Session()
    resp = session.get('https://pahar.in/')
    resp.raise_for_status()  # Ensure we got a valid response
    html = get_border_page(session)
    dom = BeautifulSoup(html, 'html.parser')
    links = dom.find_all('a', href=True)
    for link in links:
        link_text = link.get_text(strip=True)
        if link_text.startswith('Sheet '):
            href = link['href']
            sheet_no = link_text.split(' ')[1]
            jpg_file = Path(f'data/raw/border/sheet_{sheet_no}.jpg')
            if not jpg_file.exists():
                print(f"Downloading {sheet_no} from {href}")
                jpg_resp = session.get(href)
                jpg_resp.raise_for_status()
                jpg_file.parent.mkdir(parents=True, exist_ok=True)
                jpg_file.write_bytes(jpg_resp.content)
            else:
                print(f"File {jpg_file} already exists, skipping download.")

