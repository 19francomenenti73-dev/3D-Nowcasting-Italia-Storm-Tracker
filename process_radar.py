import json
import requests
import numpy as np
import cv2
import os
import sys
from io import BytesIO
from PIL import Image, ImageDraw
from datetime import datetime

os.makedirs("profiles", exist_ok=True)

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
            return host, latest.get("path")
    except Exception as e:
        print(f"Avviso nel recupero radar: {e}")
    return "https://tilecache.rainviewer.com", "/v2/radar/1710000000"

def save_iso_profile_image(grid_data, filename):
    try:
        img = Image.new("RGBA", (160, 95), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if grid_data and len(grid_data) > 0:
            rows = len(grid_data)
            cols = len(grid_data[0])
            tileW = 8
            tileH = 4
            startX = 80
            startY = 10

            def get_color(val):
                if val >= 12: return (255, 0, 255, 250)      # Magenta (Picco estremo)
                elif val >= 10: return (255, 26, 26, 250)   # Rosso (Forte)
                elif val >= 8: return (255, 204, 0, 250)    # Giallo (Moderato-Alto)
                elif val >= 6: return (0, 230, 0, 250)      # Verde (Base moderata)
                elif val >= 4: return (0, 191, 255, 250)    # Ciano (Debole)
                elif val > 0: return (0, 128, 255, 250)     # Blu (Leggero)
                return None

            for r in range(rows):
                for c in range(cols):
                    val = grid_data[r][c]
                    if val >= 6:
                        isoX = startX + (c - r) * (tileW / 2)
                        isoY = startY + (c + r) * (tileH / 2)
                        color = get_color(val)
                        if color:
                            for h in range(val):
                                hY = isoY - (h * 2.8)
                                draw.ellipse([isoX - 3, hY - 3, isoX + 3, hY + 3], fill=color)

        img.save(filename, format="PNG")
    except Exception as e:
        print(f"Errore generazione immagine profilo {filename}: {e}")

def create_fallback_data(reason="Standby"):
    default_id = "Core-Standby-01"
    default_img = f"profiles/{default_id}.png"
    save_iso_profile_image([[0]*15 for _ in range(15)], default_img)
    
    data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radar_tile": {
            "host": "https://tilecache.rainviewer.com",
            "path": "/v2/radar/1710000000"
        },
        "macro_structures": [
            {
                "id": default_id,
                "center": [45.0, 10.0],
                "speed_kmh": 40,
                "direction_deg": 45,
                "intensity": f"Sistema operativo ({reason})",
                "vil": 0.0,
                "echo_top": 0.0,
                "profile_image": default_img,
                "actual_path": [[44.8, 9.8], [44.9, 9.9], [45.0, 10.0]],
                "forecast_path": [[45.2, 10.2], [45.4, 10.4], [45.6, 10.6]]
            }
        ]
    }
    with open("storm_cells.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def analyze_radar():
    try:
        host, path = get_latest_radar_tile_info()
        radar_info = {"host": host, "path": path}
        macro_structures = []
        
        z = 4
        tiles_to_check = []
        for x in range(6, 11):
            for y in range(3, 8):
                tiles_to_check.append((x, y))

        cell_id_counter = 1

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
                    
                    mask_precipitation = (alpha > 80) & ((r > 130) | (g > 180)) & (b < 200)
                    if not np.any(mask_precipitation):
                        continue

                    kernel = np.ones((2,2), np.uint8)
                    mask_clean = cv2.morphologyEx(mask_precipitation.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
                    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 10:
                            x_c, y_c, w, h = cv2.boundingRect(cnt)
                            lat, lon = tile_pixel_to_latlon(z, x, y, x_c + w / 2.0, y_c + h / 2.0)
                            
                            # Bounding box esteso a tutta Europa
                            if 35.0 <= lat <= 60.0 and -10.0 <= lon <= 30.0:
                                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                                
                                if aspect_ratio > 3.0:
                                    classification = "MCS / Linea di Groppo"
                                elif aspect_ratio > 1.8:
                                    classification = "Bow Echo"
                                elif area > 250:
                                    classification = "MCC"
                                else:
                                    classification = "Supercella / Cella"

                                vil_val = round(min(70.0, 10.0 + (area * 0.18)), 1)
                                echo_top_val = round(min(16.0, 7.0 + (area * 0.035)), 1)
                                speed_val = int(35 + (area % 30))
                                direction_deg = int((lat * 22 + lon * 18) % 360)
                                
                                rad_dir = np.radians(direction_deg)
                                step_dist = speed_val * 0.00035
                                lat_dir = np.cos(rad_dir)
                                lon_dir = np.sin(rad_dir)

                                actual_path = [
                                    [lat - lat_dir * step_dist * 3, lon - lon_dir * step_dist * 3],
                                    [lat - lat_dir * step_dist * 2, lon - lon_dir * step_dist * 2],
                                    [lat - lat_dir * step_dist * 1, lon - lon_dir * step_dist * 1],
                                    [lat, lon]
                                ]

                                # Linee previsionali più lunghe (moltiplicatori 8, 16, 24)
                                forecast_path = [
                                    [lat + lat_dir * step_dist * 8, lon + lon_dir * step_dist * 8],
                                    [lat + lat_dir * step_dist * 16, lon + lon_dir * step_dist * 16],
                                    [lat + lat_dir * step_dist * 24, lon + lon_dir * step_dist * 24]
                                ]

                                patch_size = 15
                                half_p = patch_size // 2
                                px_center = int(x_c + w / 2.0)
                                py_center = int(y_c + h / 2.0)
                                
                                x_min = max(0, px_center - half_p)
                                x_max = min(arr.shape[1], px_center + half_p + 1)
                                y_min = max(0, py_center - half_p)
                                y_max = min(arr.shape[0], py_center + half_p + 1)
                                
                                local_patch = arr[y_min:y_max, x_min:x_max]
                                grid_matrix = []
                                for row in local_patch:
                                    row_vals = []
                                    for pixel in row:
                                        pr, pg, pb, pa = pixel[0], pixel[1], pixel[2], pixel[3]
                                        if pa < 50:
                                            row_vals.append(0)
                                        else:
                                            if pr > 200 and pb > 200: row_vals.append(12)
                                            elif pr > 200 and pg < 100: row_vals.append(10)
                                            elif pr > 200 and pg > 150: row_vals.append(8)
                                            elif pg > 200: row_vals.append(6)
                                            elif pb > 200 and pg > 150: row_vals.append(4)
                                            elif pb > 150: row_vals.append(2)
                                            else: row_vals.append(1)
                                    grid_matrix.append(row_vals)

                                track_id = f"Core-{z}{x}{y}-{cell_id_counter}"
                                img_filename = f"profiles/{track_id}.png"
                                save_iso_profile_image(grid_matrix, img_filename)

                                data_item = {
                                    "id": track_id,
                                    "center": [lat, lon],
                                    "speed_kmh": speed_val,
                                    "direction_deg": direction_deg,
                                    "intensity": f">= 32 dBZ — {classification}",
                                    "vil": vil_val,
                                    "echo_top": echo_top_val,
                                    "profile_image": img_filename,
                                    "actual_path": actual_path,
                                    "forecast_path": forecast_path
                                }
                                
                                if not any(abs(c["center"][0] - lat) < 0.12 and abs(c["center"][1] - lon) < 0.12 for c in macro_structures):
                                    macro_structures.append(data_item)
                                    cell_id_counter += 1
            except Exception as tile_err:
                print(f"Nota tile: {tile_err}")

        if not macro_structures:
            create_fallback_data("Nessun nucleo intenso")
        else:
            data = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "radar_tile": radar_info,
                "macro_structures": macro_structures
            }
            with open("storm_cells.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Errore generale: {e}")
        create_fallback_data("Ripristino")

if __name__ == "__main__":
    analyze_radar()
    sys.exit(0)
                                
