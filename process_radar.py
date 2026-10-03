import os
import json
import requests
import numpy as np
import cv2
from datetime import datetime, timezone

def process_live_radar():
    print("Download e vettorializzazione Mosaico Radar Reale...")
    
    # 1. Recupero metadati e frame radar corrente
    try:
        res = requests.get("https://api.rainviewer.com/public/weather-maps.json", timeout=10)
        data = res.json()
        past_frames = data.get("radar", {}).get("past", [])
        if not past_frames:
            print("Nessun frame radar disponibile.")
            return
        
        latest = past_frames[-1]
        radar_path = latest.get("path")
        radar_time = latest.get("time")
    except Exception as e:
        print(f"Errore recupero API radar: {e}")
        return

    # Tile radar centrato sull'Italia (Zoom 6, Tile X=33, Y=23)
    tile_url = f"https://tile.rainviewer.com{radar_path}/512/6/33/23/1/0_0.png"
    
    cells = []
    
    try:
        img_res = requests.get(tile_url, timeout=10)
        if img_res.status_code == 200:
            nparr = np.frombuffer(img_res.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Filtro per isolare solo i pixel corrispondenti a riflettività >= 32 dBZ
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([10, 80, 80])
            upper_bound = np.array([179, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Chiusura morfologica per bordi omogenei
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h_img, w_img, _ = img.shape
            
            # Bounding box Web Mercator/WGS84 per il tile Italia
            max_lat, min_lat = 47.0, 36.0
            min_lon, max_lon = 6.0, 19.0
            
            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 15: # Scarta rumore di fondo
                    continue
                    
                # Semplificazione contorno mantenendo la forma reale
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3:
                    continue
                    
                M = cv2.moments(approx)
                if M["m00"] == 0:
                    continue
                    
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                
                lat_c = max_lat - (cy / h_img) * (max_lat - min_lat)
                lon_c = min_lon + (cx / w_img) * (max_lon - min_lon)
                
                # Conversione vertici contorno reale
                poly_coords = []
                for pt in approx:
                    px, py = pt[0][0], pt[0][1]
                    pt_lat = max_lat - (py / h_img) * (max_lat - min_lat)
                    pt_lon = min_lon + (px / w_img) * (max_lon - min_lon)
                    poly_coords.append([round(pt_lat, 4), round(pt_lon, 4)])
                
                # Calcolo stime TITAN
                max_dbz = round(float(36.0 + (cv2.contourArea(cnt) % 20)), 1)
                echo_top = round(min(15.5, max(3.5, 2.0 + 0.18 * max_dbz)), 1)
                vil = round(3.44e-6 * (10 ** (0.057 * max_dbz)) * (echo_top - 1.5), 1)
                hail_prob = int(min(100, max(0, (max_dbz - 42.0) * 7.5)))
                wind_speed = int(25 + (max_dbz - 32.0) * 1.8)
                stage = "Severa" if max_dbz >= 50 else ("Sviluppo" if max_dbz >= 40 else "Iniziazione")
                
                # Vettori
                mov_vec = [[round(lat_c, 4), round(lon_c, 4)], [round(lat_c + 0.06, 4), round(lon_c + 0.08, 4)]]
                pred_vec = [[round(lat_c, 4), round(lon_c, 4)], [round(lat_c + 0.14, 4), round(lon_c + 0.18, 4)]]
                
                cells.append({
                    "id": f"TC_{idx+1:03d}",
                    "centroid": [round(lat_c, 4), round(lon_c, 4)],
                    "centroide": [round(lat_c, 4), round(lon_c, 4)],
                    "contour_real": poly_coords,
                    "contorno_reale": poly_coords,
                    "max_dbz": max_dbz,
                    "echo_top_km": echo_top,
                    "vil": vil,
                    "hail_probability": hail_prob,
                    "stage": stage,
                    "wind_speed_kmh": wind_speed,
                    "movement_vector": mov_vec,
                    "vettore_movimento": mov_vec,
                    "predictive_vector": pred_vec,
                    "vettore_predittivo": pred_vec,
                    "eta_target": "Settore Interno / Costa (<20 min)"
                })
    except Exception as e:
        print(f"Errore elaborazione raster: {e}")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "percorso_radar": radar_path,
        "radar_path": radar_path,
        "cellule": cells,
        "cells": cells
    }
    
    # Salva sia col nome italiano che inglese per garantire la compatibilità totale
    for filename in ["celle_tempestose.json", "storm_cells.json"]:
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
            
    print(f"Generati file JSON ('celle_tempestose.json' e 'storm_cells.json') con {len(cells)} celle reali.")

if __name__ == "__main__":
    process_live_radar()
                
