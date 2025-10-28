import argparse
import cv2
import yaml
from detectors.yolov8_detector import YOLOv8Detector
from trackers.bytetrack import ByteTrackWrapper
from utils.visualize import draw_tracks, draw_associations
from utils.comovement_detector import CoMovementDetector

def run(video_path, output_path, config_path):
    print(f"[INFO] Cargando configuración desde: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Cargando modelo YOLOv8...")

    # Parse class grouping configuration if available
    class_grouping = config.get("class_grouping", {})
    if class_grouping:
        print(f"[INFO] Configuración de agrupamiento de clases cargada")

    # Parse class-specific parameters if available
    class_conf_thresholds = {}
    class_min_box_areas = {}
    default_conf = config.get("conf", 0.25)  # Fallback for old config format

    if "class_parameters" in config:
        class_params = config["class_parameters"]
        default_params = class_params.get("default", {})
        default_conf = default_params.get("conf", 0.25)
        default_min_box_area = default_params.get("min_box_area", 0)

        # Build per-class parameter dictionaries
        for cls in config["detect_classes"]:
            if cls in class_params:
                class_conf_thresholds[cls] = class_params[cls].get("conf", default_conf)
                class_min_box_areas[cls] = class_params[cls].get("min_box_area", default_min_box_area)
            else:
                class_conf_thresholds[cls] = default_conf
                class_min_box_areas[cls] = default_min_box_area

        print(f"[INFO] Usando configuración por clase:")
        print(f"  - Umbrales de confianza: {class_conf_thresholds}")
        print(f"  - Áreas mínimas de caja: {class_min_box_areas}")

    detector = YOLOv8Detector(
        model_path=config["model_path"],
        detect_classes=config["detect_classes"],
        conf=default_conf,
        iou=config["iou"],
        class_conf_thresholds=class_conf_thresholds if class_conf_thresholds else None,
        class_min_box_areas=class_min_box_areas if class_min_box_areas else None,
        class_grouping=class_grouping if class_grouping else None
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

    # Initialize co-movement detector if enabled
    comovement_config = config.get("comovement_detection", {})
    comovement_detector = None

    if comovement_config.get("enable", False):
        from utils.visualize import CLASS_NAMES

        print(f"[INFO] Detección de co-movimiento habilitada")
        print(f"[INFO] Umbral de proximidad: {comovement_config.get('proximity_threshold', 150)} pixels")
        print(f"[INFO] Frames mínimos: {comovement_config.get('min_frames', 10)}")

        comovement_detector = CoMovementDetector(
            proximity_threshold=comovement_config.get('proximity_threshold', 150),
            min_frames=comovement_config.get('min_frames', 10),
            max_gap_frames=comovement_config.get('max_gap_frames', 5),
            class_names=CLASS_NAMES
        )

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

        # Update co-movement detector if enabled
        associations = []
        if comovement_detector is not None:
            associations = comovement_detector.update(tracks, frame_idx)

        # Get trajectory visualization configuration
        trajectory_viz_config = config.get("trajectory_visualization", {})

        # Draw tracks with trajectories
        frame = draw_tracks(
            frame,
            tracks,
            tracker=tracker,
            draw_trajectories=trajectory_viz_config.get("enable", True),
            trajectory_tail_length=trajectory_viz_config.get("tail_length", 30),
            trajectory_thickness=trajectory_viz_config.get("thickness", 2),
            trajectory_fade=trajectory_viz_config.get("fade", True)
        )

        # Draw associations if enabled
        if associations and comovement_config.get("enable", False):
            viz_config = comovement_config.get("visualization", {})
            frame = draw_associations(
                frame,
                associations,
                draw_connections=viz_config.get("draw_connections", True),
                connection_color=tuple(viz_config.get("connection_color", [0, 255, 0])),
                connection_thickness=viz_config.get("connection_thickness", 2),
                highlight_boxes=viz_config.get("highlight_boxes", True),
                show_labels=viz_config.get("show_labels", True)
            )

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