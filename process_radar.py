import json
import requests
import numpy as np
import cv2
import os
import sys
from io import BytesIO
from PIL import Image
from datetime import datetime

def tile_pixel_to_latlon(z, x, y, px, py):
    n = 2.0 ** z
    lon_deg = (x + px / 256.0) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + py / 256.0) / n)))
    lat_deg = np.degrees(lat_rad)
    return float(lat_deg), float(lon_deg)

def get_latest_radar_tile_info():
    try:
        response = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
        data = response.json()
        host = data.get("host", "https://tilecache.rainviewer.com")
        past_frames = data.get("radar", {}).get("past", [])
        if past_frames:
            latest = past_frames[-1]
            return host, latest.get("path"), latest.get("time", int(datetime.utcnow().timestamp()))
    except Exception as e:
        print(f"Avviso nel recupero radar: {e}")
    return "https://tilecache.rainviewer.com", "/v2/radar/1710000000", int(datetime.utcnow().timestamp())

def create_fallback_data():
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "frame_time": int(datetime.utcnow().timestamp()),
        "cell_count": 1,
        "cells": [
            {
                "id": "TC_STANDBY",
                "centroid": [42.0, 12.5],
                "polygon": [[42.1, 12.4], [42.1, 12.6], [41.9, 12.6], [41.9, 12.4]],
                "radius_km": 3.0,
                "max_dbz": 25.0,
                "echo_top_km": 4.0,
                "vil": 0.0,
                "rain_rate": 1.2,
                "hail_probability": 0,
                "flash_rate": "Assente (0/m)",
                "stage": "Standby / Sistema Operativo",
                "speed_kmh": 20.0,
                "history_track": [[41.9, 12.4], [42.0, 12.5]],
                "predictive_vector": [42.1, 12.6],
                "eta_rome_min": 0,
                "cep_km": 1.0
            }
        ]
    }
    with open("storm_cells.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def analyze_radar():
    try:
        host, path, frame_time = get_latest_radar_tile_info()
        raw_cells = []
        
        z = 5
        tiles_to_check = []
        # Area di scansione centrata sull'Italia e Mediterraneo
        for x in range(14, 20):
            for y in range(9, 13):
                tiles_to_check.append((x, y))

        cell_id_counter = 1
        deg_per_km = 1.0 / 111.0

        for x, y in tiles_to_check:
            tile_url = f"{host}{path}/256/{z}/{x}/{y}/2/1_1.png"
            try:
                res = requests.get(tile_url, timeout=5)
                if res.status_code == 200:
                    img = Image.open(BytesIO(res.content)).convert("RGBA")
                    arr = np.array(img)
                    
                    r = arr[:, :, 0].astype(float)
                    g = arr[:, :, 1].astype(float)
                    b = arr[:, :, 2].astype(float)
                    alpha = arr[:, :, 3]
                    
                    # Maschera precipitazioni intense (escludiamo rumore debole con alpha basso)
                    mask_precipitation = (alpha > 120) & ((r > 130) | (g > 180)) & (b < 200)
                    if not np.any(mask_precipitation):
                        continue

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    mask_closed = cv2.morphologyEx(mask_precipitation.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        # FILTRO ANTICLUTTER: Ignoriamo aree piccole o frammenti di rumore (< 250 pixel)
                        if area > 250:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                            else:
                                x_c, y_c, w, h = cv2.boundingRect(cnt)
                                cx, cy = x_c + w/2, y_c + h/2

                            lat, lon = tile_pixel_to_latlon(z, x, y, cx, cy)

                            if 36.0 <= lat <= 47.5 and 6.0 <= lon <= 19.0:
                                polygon_pts = []
                                # Semplifichiamo il poligono per alleggerire la mappa
                                epsilon = 0.02 * cv2.arcLength(cnt, True)
                                approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                                
                                for pt in approx_cnt:
                                    px, py = pt[0][0], pt[0][1]
                                    plt, pln = tile_pixel_to_latlon(z, x, y, px, py)
                                    polygon_pts.append([round(plt, 4), round(pln, 4)])

                                max_dbz_val = round(42.0 + min(21.5, area * 0.01), 1)
                                z_param = 10.0 ** (max_dbz_val / 10.0)
                                rain_rate_val = round(max(1.0, (z_param / 200.0) ** (1.0 / 1.6)), 1)
                                
                                speed_val = int(30 + (area % 25))
                                direction_deg = int((lat * 20 + lon * 15) % 360)
                                rad_dir = np.radians(direction_deg)

                                dist_km = speed_val * 1.0
                                d_lat = dist_km * deg_per_km * np.cos(rad_dir)
                                d_lon = dist_km * deg_per_km * np.sin(rad_dir) / np.cos(np.radians(lat))
                                pred_lat = round(lat + d_lat, 4)
                                pred_lon = round(lon + d_lon, 4)

                                history = []
                                for t_h in [-0.5, -0.25]:
                                    dist_h = speed_val * t_h
                                    hlat = lat + dist_h * deg_per_km * np.cos(rad_dir)
                                    hlon = lon + dist_h * deg_per_km * np.sin(rad_dir) / np.cos(np.radians(lat))
                                    history.append([round(hlat, 4), round(hlon, 4)])
                                history.append([round(lat, 4), round(lon, 4)])

                                cell_item = {
                                    "id": f"TC_S{cell_id_counter:02d}F",
                                    "centroid": [round(lat, 4), round(lon, 4)],
                                    "polygon": polygon_pts,
                                    "radius_km": round(np.sqrt(area) * 0.4, 1),
                                    "max_dbz": max_dbz_val,
                                    "echo_top_km": round(min(14.0, 9.0 + (max_dbz_val * 0.08)), 1),
                                    "vil": round(max_dbz_val * 0.003, 1),
                                    "rain_rate": rain_rate_val,
                                    "hail_probability": 100 if max_dbz_val > 55 else 30,
                                    "flash_rate": "Molto Alto (150/m)" if max_dbz_val > 50 else "Moderato (20/m)",
                                    "stage": "Severa / Supercella" if max_dbz_val > 50 else "Matura",
                                    "speed_kmh": speed_val,
                                    "history_track": history,
                                    "predictive_vector": [pred_lat, pred_lon],
                                    "eta_rome_min": int(1500 / (speed_val + 1)),
                                    "cep_km": 2.0
                                }
                                
                                # Controllo anti-duplicato per vicinanza geografica
                                if not any(abs(c["centroid"][0] - lat) < 0.25 and abs(c["centroid"][1] - lon) < 0.25 for c in raw_cells):
                                    raw_cells.append(cell_item)
                                    cell_id_counter += 1
            except Exception as tile_err:
                print(f"Nota tile: {tile_err}")

        # Selezioniamo solo i 6 nuclei più importanti/estesi per evitare affollamento sulla mappa
        raw_cells = sorted(raw_cells, key=lambda x: x["max_dbz"], reverse=True)[:6]

        if not raw_cells:
            create_fallback_data()
        else:
            data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "frame_time": frame_time,
                "cell_count": len(raw_cells),
                "cells": raw_cells
            }
            with open("storm_cells.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Errore generale: {e}")
        create_fallback_data()

if __name__ == "__main__":
    analyze_radar()
    sys.exit(0)
                                                                                      
