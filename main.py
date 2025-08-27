import argparse
import cv2
import os
from detectors.yolov8_detector import YOLOv8Detector
from trackers.bytetrack import ByteTrackWrapper
from utils.visualize import draw_tracks, draw_tracks_with_trajectories
from utils.trajectory_utils import TrajectoryExporter, TrajectoryAnalyzer

def run(video_path, output_path, save_trajectories=True, show_trajectories=True):
    print(f"[INFO] Cargando modelo YOLOv8...")
    detector = YOLOv8Detector("yolov8m.pt")

    print(f"[INFO] Inicializando ByteTrack...")
    tracker = ByteTrackWrapper()

    print(f"[INFO] Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    active_track_ids = set()
    
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
        
        # Track active track IDs
        current_track_ids = set(track[4] for track in tracks)
        active_track_ids.update(current_track_ids)

        # Use enhanced trajectory visualization if enabled
        if show_trajectories:
            frame = draw_tracks_with_trajectories(frame, tracks, tracker, trail_length=50)
        else:
            frame = draw_tracks(frame, tracks)
        out.write(frame)

        if frame_idx % 50 == 0:
            print(f"[INFO] Guardados {frame_idx} frames...")
            # Show trajectory progress every 50 frames
            trajectories = tracker.get_track_trajectories()
            if trajectories:
                print(f"[TRAJ] Total tracks con trayectorias: {len(trajectories)}")
                for track_id, trajectory in trajectories.items():
                    print(f"[TRAJ]   Track {track_id}: {len(trajectory)} puntos")

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video procesado guardado en: {output_path}")
    
    # Save and analyze trajectories
    if save_trajectories:
        print(f"\n[INFO] Exportando trayectorias...")
        trajectories = tracker.get_track_trajectories()
        
        if trajectories:
            # Create output directory for trajectory files
            base_name = os.path.splitext(output_path)[0]
            trajectory_dir = f"{base_name}_trajectories"
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Export trajectories in different formats
            json_path = os.path.join(trajectory_dir, "trajectories.json")
            csv_path = os.path.join(trajectory_dir, "trajectories.csv")
            mot_path = os.path.join(trajectory_dir, "trajectories_mot.txt")
            
            TrajectoryExporter.export_to_json(trajectories, json_path)
            TrajectoryExporter.export_to_csv(trajectories, csv_path)
            TrajectoryExporter.export_to_mot_format(trajectories, mot_path)
            
            print(f"[INFO] Trayectorias exportadas a:")
            print(f"  - JSON: {json_path}")
            print(f"  - CSV: {csv_path}")
            print(f"  - MOT format: {mot_path}")
            
            # Analyze trajectories
            print(f"\n[INFO] Analizando trayectorias...")
            summary = TrajectoryAnalyzer.get_trajectory_summary(trajectories)
            
            # Print analysis results
            print(f"[ANALYSIS] Total de tracks procesados: {len(active_track_ids)}")
            print(f"[ANALYSIS] Tracks con trayectorias: {len(trajectories)}")
            print(f"\n[ANALYSIS] Resumen de trayectorias:")
            
            for track_id, stats in summary.items():
                if stats['length_points'] > 0:
                    print(f"  Track {track_id}:")
                    print(f"    - Puntos de trayectoria: {stats['length_points']}")
                    print(f"    - Duración: {stats['duration_frames']} frames ({stats['start_frame']}-{stats['end_frame']})")
                    print(f"    - Longitud de ruta: {stats['path_length_pixels']:.2f} píxeles")
                    print(f"    - Velocidad promedio: {stats['average_speed']:.2f} píxeles/frame")
                    print(f"    - Cambios de dirección: {stats['direction_changes']}")
                    print(f"    - Confianza promedio: {stats['average_confidence']:.3f}")
                    print(f"    - Límites (x_min, y_min, x_max, y_max): {stats['bounds']}")
                    print()
            
            # Save analysis summary
            summary_path = os.path.join(trajectory_dir, "trajectory_analysis.json")
            with open(summary_path, 'w') as f:
                import json
                json.dump(summary, f, indent=2)
            print(f"[INFO] Análisis guardado en: {summary_path}")
        else:
            print("[WARNING] No se encontraron trayectorias para exportar.")
    
    print(f"[INFO] Procesamiento completado. Tracks totales detectados: {len(active_track_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People tracking with trajectory storage")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--save-trajectories", action="store_true", default=True, 
                       help="Save trajectory data (enabled by default)")
    parser.add_argument("--no-trajectories", action="store_true", 
                       help="Disable trajectory saving")
    parser.add_argument("--show-trajectories", action="store_true", default=True,
                       help="Show trajectory paths in output video (enabled by default)")
    parser.add_argument("--no-trajectory-vis", action="store_true",
                       help="Disable trajectory visualization in video")
    args = parser.parse_args()

    save_trajectories = args.save_trajectories and not args.no_trajectories
    show_trajectories = args.show_trajectories and not args.no_trajectory_vis
    run(args.video, args.output, save_trajectories, show_trajectories)