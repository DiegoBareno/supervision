import argparse
from typing import List
import csv

import cv2
import numpy as np
from inference import get_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

def main(
    source_video_path: str,
    zone_configuration_path: str,
    model_id: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    model = get_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    # Conservamos el timer para dibujar, aunque para el log usaremos nuestro propio método
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # Diccionarios para guardar detecciones activas por zona
    # Cada elemento de active_intervals es un diccionario: { tracker_id: {"start": tiempo_inicio, "last_seen": tiempo_último} }
    active_intervals = [dict() for _ in zones]
    # Lista para intervalos completados: cada entrada es un diccionario con tracker_id, zona, inicio, fin y duración
    completed_intervals = []

    frame_count = 0  # Para calcular el tiempo actual en segundos
    for frame in frames_generator:
        current_time = frame_count / video_info.fps  # Tiempo en segundos
        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            # Dibujar la zona
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            # Obtener detecciones dentro de la zona
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)  # Se sigue usando para anotación
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            # Actualizamos o iniciamos el registro de detección para cada objeto en la zona
            # Primero, extraemos los tracker_id detectados en el frame actual
            current_ids = list(detections_in_zone.tracker_id)

            # Revisamos aquellos que estaban activos pero que ya no se detectan en este frame
            for tracker_id in list(active_intervals[idx].keys()):
                if tracker_id not in current_ids:
                    record = active_intervals[idx].pop(tracker_id)
                    completed_intervals.append({
                        "tracker_id": tracker_id,
                        "zone": idx,
                        "start": record["start"],
                        "end": record["last_seen"],
                        "duration": record["last_seen"] - record["start"],
                    })

            # Para cada detección actual, iniciamos o actualizamos su registro
            for tracker_id in current_ids:
                if tracker_id in active_intervals[idx]:
                    active_intervals[idx][tracker_id]["last_seen"] = current_time
                else:
                    active_intervals[idx][tracker_id] = {"start": current_time, "last_seen": current_time}

            # Anotaciones de color y etiqueta en la imagen (se muestra el tiempo acumulado)
            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # Al finalizar el video, finalizamos los intervalos activos restantes
    for idx in range(len(zones)):
        for tracker_id, record in active_intervals[idx].items():
            completed_intervals.append({
                "tracker_id": tracker_id,
                "zone": idx,
                "start": record["start"],
                "end": record["last_seen"],
                "duration": record["last_seen"] - record["start"],
            })

    cv2.destroyAllWindows()

    # Opcional: almacenar los registros en un archivo CSV
    with open("detection_logs.csv", "w", newline="") as csvfile:
        fieldnames = ["tracker_id", "zone", "start", "end", "duration"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in completed_intervals:
            writer.writerow(row)
    print("Logs de detección guardados en detection_logs.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--model_id", type=str, default="yolov8s-640", help="Roboflow model ID."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
