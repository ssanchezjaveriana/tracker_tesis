import argparse
import cv2
import yaml
from detectors.yolov8_detector import YOLOv8Detector
from trackers.bytetrack import ByteTrackWrapper
from utils.visualize import draw_tracks

def run(video_path, output_path, config_path):
    print(f"[INFO] Cargando configuración desde: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Cargando modelo YOLOv8...")
    detector = YOLOv8Detector(
        model_path=config["model_path"],
        detect_classes=config["detect_classes"]
    )

    print(f"[INFO] Inicializando ByteTrack...")
    
    # Load trajectory storage configuration if available
    trajectory_config = config.get("trajectory_storage", {})
    
    tracker = ByteTrackWrapper(
        frame_rate=30,
        track_thresh=config["track_thresh"],
        match_thresh=config["match_thresh"],
        buffer=config["track_buffer"],
        aspect_ratio_thresh=config["aspect_ratio_thresh"],
        min_box_area=config["min_box_area"],
        # Trajectory storage parameters
        enable_trajectory_storage=trajectory_config.get("enable", False),
        trajectory_output_dir=trajectory_config.get("output_dir", "data/trajectories"),
        trajectory_export_format=trajectory_config.get("export_format", "json"),
        trajectory_export_frequency=trajectory_config.get("export_frequency", 100)
    )
    
    if trajectory_config.get("enable", False):
        print(f"[INFO] Almacenamiento de trayectorias habilitado")
        print(f"[INFO] Directorio de salida: {trajectory_config.get('output_dir', 'data/trajectories')}")
        print(f"[INFO] Formato de exportación: {trajectory_config.get('export_format', 'json')}")

    print(f"[INFO] Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin del video o error de lectura.")
            break

        print(f"[INFO] Procesando frame {frame_idx}")
        detections = detector.detect(frame)
        print(f"[DEBUG] Detecciones: {len(detections)}")

        tracks = tracker.update(detections, frame)
        print(f"[DEBUG] Tracks activos: {len(tracks)}")

        frame = draw_tracks(frame, tracks)
        out.write(frame)

        if frame_idx % 10 == 0:
            print(f"[INFO] Guardados {frame_idx} frames...")

        frame_idx += 1

    # Finalize trajectory storage if enabled
    if trajectory_config.get("enable", False):
        print(f"[INFO] Finalizando almacenamiento de trayectorias...")
        tracker.finalize_trajectories()
        
        # Print trajectory summary
        summary = tracker.get_trajectory_summary()
        if summary:
            print(f"[INFO] Resumen de trayectorias:")
            print(f"  - Total de tracks: {summary['total_tracks']}")
            print(f"  - Puntos de trayectoria: {summary['total_trajectory_points']}")
            print(f"  - Frames procesados: {summary['current_frame']}")
            print(f"  - Directorio de salida: {summary['output_directory']}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video procesado guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Ruta del video de entrada")
    parser.add_argument("--output", type=str, required=True, help="Ruta del video de salida")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()

    run(args.video, args.output, args.config)