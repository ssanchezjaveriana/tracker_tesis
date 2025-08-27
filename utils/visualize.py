import cv2

# Colores RGB para cada clase
CLASS_COLORS = {
    0: (255, 0, 0),      # person - azul
    2: (0, 255, 0),      # car - verde
    24: (0, 0, 255),     # backpack - rojo
    28: (255, 255, 0),   # suitcase - cyan
    62: (255, 0, 255),   # tv - magenta
    63: (0, 255, 255),   # laptop - amarillo
    72: (100, 100, 100)  # refrigerator - gris
}

# Para clases desconocidas (por si agregas más)
DEFAULT_COLOR = (255, 255, 255)

# Para mapear los IDs de clase a nombres (opcional, para mostrar)
CLASS_NAMES = {
    0: "person",
    2: "car",
    24: "backpack",
    28: "suitcase",
    62: "tv",
    63: "laptop",
    72: "refrigerator"
}

def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2, track_id, cls_id = track
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        label = f"{CLASS_NAMES.get(cls_id, 'class')} ID:{track_id}"
        
        # Caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Etiqueta
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame