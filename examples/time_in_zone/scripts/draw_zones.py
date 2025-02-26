import argparse
import json
import os
import yt_dlp
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position: Optional[Tuple[int, int]] = None

def get_video_url(source_path: str) -> str:
    """Obtiene un enlace reproducible si es un video de YouTube."""
    if "youtube.com" in source_path or "youtu.be" in source_path:
        ydl_opts = {"format": "best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(source_path, download=False)
            return info_dict["url"]
    return source_path

def resolve_source(source_path: str) -> Optional[np.ndarray]:
    """Resuelve la fuente del video o imagen."""
    source_path = get_video_url(source_path)

    # Si es un enlace M3U8 o de red, abrir con OpenCV
    if source_path.startswith(("http://", "https://", "rtsp://")):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print("⚠️ No se pudo abrir el stream desde la URL.")
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    # Si es una imagen local
    if os.path.exists(source_path):
        image = cv2.imread(source_path)
        if image is not None:
            return image

        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        frame = next(frame_generator)
        return frame

    print("⚠️ Error: No se pudo cargar la fuente.")
    return None

def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))

def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = COLORS.by_idx(idx).as_bgr() if idx < len(POLYGONS) - 1 else sv.Color.WHITE.as_bgr()
        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(image, polygon[i - 1], polygon[i], color, THICKNESS)
            if idx < len(POLYGONS) - 1:
                cv2.line(image, polygon[-1], polygon[0], color, THICKNESS)
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(image, polygon[-1], current_mouse_position, color, THICKNESS)
    cv2.imshow(WINDOW_NAME, image)

def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(image, POLYGONS[-1][-1], POLYGONS[-1][0], COLORS.by_idx(0).as_bgr(), THICKNESS)
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)

def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(image, polygon[i], polygon[i + 1], color, THICKNESS)
            cv2.line(image, polygon[-1], polygon[0], color, THICKNESS)

def save_polygons_to_json(polygons, target_path):
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)

def main(source_path: str, zone_configuration_path: str) -> None:
    global current_mouse_position
    original_image = resolve_source(source_path)
    if original_image is None:
        print("⚠️ No se pudo cargar la imagen o video.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [KEY_ENTER, KEY_NEWLINE]:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, zone_configuration_path)
            print(f"📁 Polígonos guardados en {zone_configuration_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dibujar zonas en imágenes/videos y guardarlas en JSON.")
    parser.add_argument("--source_path", type=str, required=True, help="Ruta del archivo de imagen o video.")
    parser.add_argument("--zone_configuration_path", type=str, required=True, help="Ruta del archivo JSON de zonas.")
    arguments = parser.parse_args()
    
    main(arguments.source_path, arguments.zone_configuration_path)
