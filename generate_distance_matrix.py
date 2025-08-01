# generate_distance_matrix.py

import requests
import pandas as pd
import os
import logging
from pyproj import Transformer
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
API_KEY = os.getenv("ORS_API_KEY")
print(f"API_KEY loaded: {API_KEY}")

CSV_FILE_PATH = 'data/data.csv'
OUTPUT_PATH = 'output/distance_matrix.csv'
MAX_LOCATIONS = 50  # 50x50 = 2500 < 3500 (ORS limit)

def load_coordinates(csv_file_path):
    data = pd.read_csv(csv_file_path)

    required_cols = ['X', 'Y', 'ADDRESS']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    coords = [transformer.transform(x, y) for x, y in zip(data['X'], data['Y'])]

    addresses = data['ADDRESS'].tolist()

    # Przycinanie do maksymalnej liczby lokalizacji (ORS API limit)
    if len(coords) > MAX_LOCATIONS:
        logger.warning(f"Input has {len(coords)} locations; sampling down to {MAX_LOCATIONS} to meet ORS limits.")
        coords = coords[:MAX_LOCATIONS]
        addresses = addresses[:MAX_LOCATIONS]

    return coords, addresses

def get_ors_matrix(locations, api_key):
    if not api_key:
        logger.error("ORS_API_KEY is not set. Please check your .env file.")
        return None

    url = 'https://api.openrouteservice.org/v2/matrix/driving-car'
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        'locations': locations,
        'metrics': ['duration'],
        'units': 'm'
    }

    print("DEBUG: Sending POST request to:", url)
    print("DEBUG: Headers:", headers)
    print("DEBUG: Sample locations (first 2):", payload["locations"][:2])

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            print("DEBUG: Status code:", response.status_code)
            print("DEBUG: Response text:", response.text)
            response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from ORS: {e}")
        return None

def save_distance_matrix(matrix, labels):
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(matrix, columns=labels, index=labels)
    df.to_csv(OUTPUT_PATH)
    logger.info(f"Distance matrix saved to '{OUTPUT_PATH}'.")

def main():
    coordinates, addresses = load_coordinates(CSV_FILE_PATH)
    response = get_ors_matrix(coordinates, API_KEY)

    if response and 'durations' in response:
        matrix = response['durations']
        save_distance_matrix(matrix, addresses)
    else:
        logger.warning("Failed to retrieve or process distance matrix.")

if __name__ == '__main__':
    main()
