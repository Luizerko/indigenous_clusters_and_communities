import requests
import json
import csv
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm

# Defining base URLs
geojson_url_template = "https://terrasindigenas.org.br/isa_services/arcgisproxy/proxy?https://geo.socioambiental.org/webadaptor1/rest/services/monitoramento/arps/MapServer/0/query?returnGeometry=true&where=id_arp%3D{}%20&outSR=4326&outFields=*&inSr=4326&geometry=%7B%22xmin%22%3A-75%2C%22ymin%22%3A-30%2C%22xmax%22%3A-30%2C%22ymax%22%3A8%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D&geometryType=esriGeometryEnvelope&spatialRel=esriSpatialRelIntersects&geometryPrecision=6&f=geojson"
html_url_template = "https://terrasindigenas.org.br/pt-br/terras-indigenas/{}"

# Open CSV file to write results
output_file = "data/terras_indigenas_geolocation_raw.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "coordenadas", "povo"])

    # Looping through IDs that we know contain data
    for id_arp in tqdm(range(3571, 4050), desc=f"Requesting IDs", leave=True, total=4050-3571, ncols=100):
        # Requesting GeoJSON data
        geojson_url = geojson_url_template.format(id_arp)
        response = requests.get(geojson_url)

        coordinates = None
        if response.status_code == 200:
            try:
                data = response.json()
                features = data.get("features", [])
                for feature in features:
                    if feature["properties"].get("tipo_aps") == "TI":
                        coordinates = feature["geometry"].get("coordinates")
                        break 
            except json.JSONDecodeError:
                print(f"Error decoding JSON for ID {id_arp}")

        # Requesting HTML data for 'povo' name
        html_url = html_url_template.format(id_arp)
        response = requests.get(html_url)

        title_text = None
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.text.split(" | ")[0]

        if coordinates is not None and coordinates != '' and title_text is not None and title_text != '':
            writer.writerow([id_arp, coordinates, title_text])

        # Sleeping to avoid overloading the server
        sleep(0.1)

print(f"Finished mining data!")