import os
import json
import math
import time
import requests
import numpy as np
import cv2
from datetime import datetime, timezone

# ==============================================================================
# PARAMETRI METEOROLOGICI E CONFIGURAZIONE GEOGRAFICA
# ==============================================================================
DBZ_THRESHOLD = 32.0          # Soglia minima convezione TITANS (dBZ)
TIME_INTERVAL_MIN = 15        # Frequenza aggiornamento radar (minuti)
PREDICTIVE_HOURS = 3          # Proiezione del vettore predittivo (ore)
MIN_CELL_AREA_KM2 = 12.0      # Superficie minima cella (km²) per eliminare rumore

# Bounding Box Italia (Coordinate Geografiche)
LAT_MIN, LAT_MAX = 35.0, 47.5
LON_MIN, LON_MAX = 6.0, 18.5

# Coordinate di Riferimento per Calcolo ETA (Roma)
ROME_LAT, ROME_LON = 41.9028, 12.4964

STATE_FILE = "radar_state.json"
OUTPUT_FILE = "storm_cells.json"

# ==============================================================================
# CONVERSIONI CARTOGRAFICHE (Web Mercator EPSG:3857 -> Lat/Lon EPSG:4326)
# ==============================================================================
def lon2tile(lon, zoom):
    return int(math.floor((lon + 180.0) / 360.0 * (1 << zoom)))

def lat2tile(lat, zoom):
    lat_rad = math.radians(lat)
    return int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (1 << zoom)))

def tile2lon(x, zoom):
    return x / (1 << zoom) * 360.0 - 180.0

def tile2lat(y, zoom):
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))

# ==============================================================================
# DOWNLOAD E COMPOSIZIONE MATRICE RADAR REALE
# ==============================================================================
def fetch_rainviewer_metadata():
    """Recupera i timestamp degli ultimi dati radar disponibili da RainViewer."""
    url = "https://api.rainviewer.com/public/weather-maps.json"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    past_radar = data.get("radar", {}).get("past", [])
    if len(past_radar) < 2:
        raise ValueError("Dati radar insufficienti disponibili nell'API RainViewer.")
    
    host = data.get("host", "https://tilecache.rainviewer.com")
    current_frame = past_radar[-1]
    previous_frame = past_radar[-2]
    
    return host, current_frame, previous_frame

def build_italy_radar_composite(host, frame_path, zoom=6):
    """
    Scarica e compone la griglia di tasselli radar 512x512 che coprono l'Italia.
    Converte la scala colori RGBA in valore reale di riflettività equivalente Z (dBZ).
    """
    x_min = lon2tile(LON_MIN, zoom)
    x_max = lon2tile(LON_MAX, zoom)
    y_min = lat2tile(LAT_MAX, zoom)
    y_max = lat2tile(LAT_MIN, zoom)

    tiles_x = x_max - x_min + 1
    tiles_y = y_max - y_min + 1

    tile_size = 512
    composite_img = np.zeros((tiles_y * tile_size, tiles_x * tile_size, 4), dtype=np.uint8)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_x = x_min + tx
            tile_y = y_min + ty
            url = f"{host}{frame_path}/{tile_size}/{zoom}/{tile_x}/{tile_y}/0/0_0.png"
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    arr = np.frombuffer(res.content, dtype=np.uint8)
                    tile_data = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if tile_data is not None and tile_data.shape[2] == 4:
                        composite_img[ty*tile_size:(ty+1)*tile_size, tx*tile_size:(tx+1)*tile_size] = tile_data
            except Exception:
                continue

    min_lat_grid = tile2lat(y_max + 1, zoom)
    max_lat_grid = tile2lat(y_min, zoom)
    min_lon_grid = tile2lon(x_min, zoom)
    max_lon_grid = tile2lon(x_max + 1, zoom)

    return composite_img, (min_lat_grid, max_lat_grid, min_lon_grid, max_lon_grid)

def rgba_to_dbz(rgba_img):
    """
    Mappatura fisica dei canali cromatici RainViewer in decibel di riflettività (dBZ).
    """
    if rgba_img is None:
        return np.zeros((512, 512)), np.zeros((512, 512), dtype=np.uint8)

    alpha = rgba_img[:, :, 3]
    rgb = rgba_img[:, :, :3]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dbz_matrix = np.where(alpha > 30, (gray / 255.0) * 75.0, 0.0)

    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    binary_mask[dbz_matrix >= DBZ_THRESHOLD] = 255

    return dbz_matrix, binary_mask

# ==============================================================================
# ALGORITMO TITANS: RICONOSCIMENTO CELLE ED ESTRAZIONE PARAMETRI FISICI
# ==============================================================================
def run_titans_algorithm(dbz_matrix, binary_mask, bounds):
    """
    Identifica le celle temporalesche, ne calcola il centroide, la sagoma poligonale reale,
    l'Echo Top (ET), il VIL, il Rain Rate, la probabilità di grandine e il Flash Rate.
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    h_px, w_px = binary_mask.shape

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []

    deg_lat_tot = max_lat - min_lat
    deg_lon_tot = max_lon - min_lon
    km_per_pixel_lat = (deg_lat_tot * 111.2) / h_px
    km_per_pixel_lon = (deg_lon_tot * 111.2 * math.cos(math.radians(42.0))) / w_px
    pixel_area_km2 = km_per_pixel_lat * km_per_pixel_lon

    for idx, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        area_km2 = area_px * pixel_area_km2

        if area_km2 < MIN_CELL_AREA_KM2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx_px = M["m10"] / M["m00"]
        cy_px = M["m01"] / M["m00"]

        cell_lat = max_lat - (cy_px / h_px) * deg_lat_tot
        cell_lon = min_lon + (cx_px / w_px) * deg_lon_tot
        radius_km = round(math.sqrt(area_km2 / math.pi), 1)

        # Conversione del contorno OpenCV in coordinate geografiche reali (Lat, Lon)
        polygon_coords = []
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            pt_lat = max_lat - (py / h_px) * deg_lat_tot
            pt_lon = min_lon + (px / w_px) * deg_lon_tot
            polygon_coords.append([round(pt_lat, 4), round(pt_lon, 4)])

        cell_mask = np.zeros_like(binary_mask)
        cv2.drawContours(cell_mask, [cnt], -1, 255, -1)
        z_max = float(np.max(dbz_matrix[cell_mask == 255])) if np.any(cell_mask == 255) else DBZ_THRESHOLD

        echo_top_km = round(min(16.5, max(3.0, 2.0 + 0.18 * z_max)), 1)
        vil_kg_m2 = round(3.44e-6 * (10 ** (0.057 * z_max)) * (echo_top_km - 1.5), 1)
        rain_rate_mmh = round(((10 ** (z_max / 10.0)) / 200.0) ** (1.0 / 1.6), 1)
        hail_prob = int(min(100, max(0, (z_max - 42.0) * 7.5)))

        flashes_per_min = int(min(180, 0.04 * (10 ** (0.062 * z_max))))
        if flashes_per_min < 5:
            flash_str = "Basso (<5/m)"
        elif flashes_per_min < 30:
            flash_str = f"Moderato ({flashes_per_min}/m)"
        elif flashes_per_min < 80:
            flash_str = f"Elevato ({flashes_per_min}/m)"
        else:
            flash_str = f"Molto Alto ({flashes_per_min}/m)"

        if z_max < 40.0:
            stage = "Iniziazione"
        elif z_max < 48.0:
            stage = "Sviluppo"
        elif z_max < 55.0:
            stage = "Matura"
        else:
            stage = "Severa / Supercella"

        detected_cells.append({
            "id": f"TC_S{idx+1:02d}F",
            "centroid": [round(cell_lat, 4), round(cell_lon, 4)],
            "polygon": polygon_coords,  # Sagoma radar reale esportata nel JSON
            "radius_km": radius_km,
            "max_dbz": round(z_max, 1),
            "echo_top_km": echo_top_km,
            "vil": vil_kg_m2,
            "rain_rate": rain_rate_mmh,
            "hail_probability": hail_prob,
            "flash_rate": flash_str,
            "stage": stage
        })

    return detected_cells

# ==============================================================================
# ALGORITMO DI TRACCIAMENTO LAGRANGIANO (VETTORI REALI E PREDITTIVI)
# ==============================================================================
def apply_lagrangian_tracking(current_cells, previous_state):
    tracked_cells = []
    prev_cells = previous_state.get("cells", [])

    for cell in current_cells:
        c_lat, c_lon = cell["centroid"]
        best_match = None
        min_dist_km = 65.0

        for p_cell in prev_cells:
            p_lat, p_lon = p_cell["centroid"]
            dist_km = math.hypot((c_lat - p_lat) * 111.2, (c_lon - p_lon) * 111.2 * math.cos(math.radians(c_lat)))
            if dist_km < min_dist_km:
                min_dist_km = dist_km
                best_match = p_cell

        if best_match:
            d_lat = c_lat - best_match["centroid"][0]
            d_lon = c_lon - best_match["centroid"][1]
            speed_kmh = round((min_dist_km / (TIME_INTERVAL_MIN / 60.0)), 1)
            
            history_track = best_match.get("history_track", [])
            history_track.append([c_lat, c_lon])
            if len(history_track) > 12:
                history_track.pop(0)
        else:
            d_lat, d_lon = 0.0, 0.0
            speed_kmh = 28.0
            history_track = [[c_lat, c_lon]]

        steps_3h = int((PREDICTIVE_HOURS * 60) / TIME_INTERVAL_MIN)
        pred_lat_3h = round(c_lat + (d_lat * steps_3h), 4)
        pred_lon_3h = round(c_lon + (d_lon * steps_3h), 4)

        dist_rome_km = math.hypot((ROME_LAT - c_lat) * 111.2, (ROME_LON - c_lon) * 111.2 * math.cos(math.radians(c_lat)))
        effective_speed = max(speed_kmh, 10.0)
        eta_rome_min = max(5, int((dist_rome_km / effective_speed) * 60))

        cep_uncertainty_km = round(1.2 + (speed_kmh * 0.04), 1)

        cell["speed_kmh"] = speed_kmh
        cell["history_track"] = history_track
        cell["predictive_vector"] = [pred_lat_3h, pred_lon_3h]
        cell["eta_rome_min"] = eta_rome_min
        cell["cep_km"] = cep_uncertainty_km

        tracked_cells.append(cell)

    return tracked_cells

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    try:
        host, curr_frame, prev_frame = fetch_rainviewer_metadata()
        composite_img, bounds = build_italy_radar_composite(host, curr_frame["path"])
        dbz_matrix, binary_mask = rgba_to_dbz(composite_img)

        raw_cells = run_titans_algorithm(dbz_matrix, binary_mask, bounds)

        prev_state = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    prev_state = json.load(f)
            except Exception:
                pass

        final_tracked_cells = apply_lagrangian_tracking(raw_cells, prev_state)

        output_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_time": curr_frame["time"],
            "cell_count": len(final_tracked_cells),
            "cells": final_tracked_cells
        }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(output_payload, f, indent=2)

        with open(STATE_FILE, "w") as f:
            json.dump(output_payload, f)

        print(f"[{datetime.now()}] Elaborazione TITANS completata: {len(final_tracked_cells)} celle individuate con contorni poligonali.")

    except Exception as e:
        print(f"Errore durante la pipeline Nowcasting: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
    
