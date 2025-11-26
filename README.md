# Sistema de Detección y Seguimiento de Objetos con Análisis de Trayectorias

Este proyecto implementa un sistema completo de detección, seguimiento y análisis de trayectorias de objetos en video utilizando YOLOv8 y ByteTrack, con capacidades de clustering para identificar patrones de comportamiento anómalos.

## Tabla de Contenidos
- [Instalación](#instalación)
- [Uso del Sistema](#uso-del-sistema)
- [Configuración](#configuración)
- [Pipeline Completo](#pipeline-completo)
- [Análisis de Trayectorias](#análisis-de-trayectorias)

## Instalación

### 1. Crear entorno virtual
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Instalar ByteTrack
```bash
git clone https://github.com/FoundationVision/ByteTrack.git byte_track_repo
cd byte_track_repo
pip install -e .
cd ..
```

### 4. Verificar instalación
```bash
python -c "import yolox; print('YOLOX instalado correctamente.')"
```

## Uso del Sistema

### Procesamiento de Video

El script principal procesa videos aplicando detección y seguimiento de objetos:

```bash
python main.py --video data/input_videos/{video_name}.mp4 \
               --output data/output_videos/{video_name}.mp4 \
               --config configs/{config_file}.yaml
```

**Parámetros:**
- `--video`: Ruta del video de entrada
- `--output`: Ruta donde se guardará el video procesado
- `--config`: Archivo de configuración YAML (ver sección [Configuración](#configuración))

**Ejemplo:**
```bash
python main.py --video data/input_videos/street_view.mp4 \
               --output data/output_videos/street_view_tracked.mp4 \
               --config configs/unified.yaml
```

## Configuración

El proyecto incluye tres archivos de configuración en [configs/](configs/):

### [configs/unified.yaml](configs/unified.yaml)
Configuración completa con múltiples clases, agrupamiento de clases similares, detección de co-movimiento y almacenamiento de trayectorias.

**Características principales:**
- Detección de 20 clases de COCO (personas, vehículos, objetos)
- Agrupamiento de clases similares (ej: todas las bolsas → backpack)
- Parámetros específicos por clase (confianza y área mínima)
- Almacenamiento de trayectorias en JSON
- Visualización de trayectorias con efecto fade
- Detección de co-movimiento persona-objeto

### [configs/person_only.yaml](configs/person_only.yaml)
Configuración simplificada para seguimiento únicamente de personas.

### [configs/multiclass.yaml](configs/multiclass.yaml)
Configuración para detección de múltiples clases sin agrupamiento.

### Parámetros Clave

**Detección (YOLOv8):**
- `model_path`: Modelo de YOLOv8 (ej: "yolov8m.pt")
- `detect_classes`: Lista de IDs de clases COCO a detectar
- `conf`: Umbral de confianza (0.0-1.0)
- `iou`: Umbral de IoU para NMS

**Seguimiento (ByteTrack):**
- `track_thresh`: Umbral de confianza para iniciar tracks
- `match_thresh`: Umbral para asociación de detecciones
- `track_buffer`: Frames que un track puede estar inactivo
- `aspect_ratio_thresh`: Filtro de aspect ratio
- `min_box_area`: Área mínima de bounding box

**Almacenamiento de Trayectorias:**
```yaml
trajectory_storage:
  enable: true
  output_dir: "data/trajectories"
  export_format: "json"  # "json", "csv", o "both"
  export_frequency: 100  # Exportar cada N frames
```

**Visualización de Trayectorias:**
```yaml
trajectory_visualization:
  enable: true
  tail_length: 200  # Puntos históricos a mostrar
  thickness: 2
  fade: true  # Efecto de desvanecimiento
```

**Detección de Co-Movimiento:**
```yaml
comovement_detection:
  enable: true
  proximity_threshold: 100  # Distancia máxima en píxeles
  min_frames: 5  # Frames mínimos para confirmar asociación
  max_gap_frames: 15  # Frames permitidos sin proximidad
```

## Pipeline Completo

### 1. Procesamiento de Video y Extracción de Trayectorias

```bash
# Procesar video con almacenamiento de trayectorias habilitado
python main.py --video data/input_videos/video.mp4 \
               --output data/output_videos/video_tracked.mp4 \
               --config configs/unified.yaml
```

**Salidas:**
- Video procesado con visualizaciones: `data/output_videos/video_tracked.mp4`
- Trayectorias en formato JSON: `data/trajectories/*.json`

### 2. Exportar Trayectorias a CSV

Usar el notebook [export_trajectories.ipynb](export_trajectories.ipynb) para consolidar todos los archivos JSON en un único CSV:

```python
# El notebook procesa por lotes para optimizar memoria
# Genera: trayectorias_completas.csv
```

**Formato del CSV:**
```
track_id, frame_id, timestamp, cx, cy
1, 523, 1761105123.45, 745.5, 213.2
```

### 3. Análisis de Trayectorias con K-Means

Usar el notebook [k_means.ipynb](k_means.ipynb) para análisis de clustering:

**Proceso:**

1. **Cálculo de Features:**
   - Deltas de posición (x_delta, y_delta)
   - Distancia euclidiana
   - Velocidad y magnitud
   - Suavizado con filtro Savitzky-Golay
   - Histogramas de distribución (10 bins por métrica)

2. **Clustering:**
   - Método del codo para determinar K óptimo
   - K-Means con k=3
   - Métricas de validación:
     - Silhouette Score
     - Davies-Bouldin Index
     - Calinski-Harabasz Index

3. **Visualización:**
   - PCA para reducción dimensional
   - Heatmaps de features por cluster
   - Análisis de importancia de features

**Modelos Generados:**
- `modelo_kmeans.joblib`: Modelo K-Means entrenado
- `scaler_kmeans.joblib`: Escalador StandardScaler
- `pca_modelo.joblib`: Modelo PCA
- `features_combinados.csv`: Features extraídas

### 4. Inferencia con Modelo Entrenado

Usar el notebook [inference.ipynb](inference.ipynb) para clasificar nuevas trayectorias:

```python
# Cargar modelos
kmeans = load('modelo_kmeans.joblib')
scaler = load('scaler_kmeans.joblib')
pca = load('pca_modelo.joblib')

# Clasificar nuevas trayectorias
cluster_labels = kmeans.predict(scaler.transform(new_features))
```

## Análisis de Trayectorias

### Features Extraídas

**Estadísticas básicas (por track):**
- `total_points`: Número de puntos en la trayectoria
- `x_delta_min/max/avg`: Estadísticas de desplazamiento horizontal
- `y_delta_min/max/avg`: Estadísticas de desplazamiento vertical
- `distance_min/max/avg`: Estadísticas de distancia recorrida
- `velocity_min/max/avg`: Estadísticas de velocidad

**Histogramas (40 features):**
- `x_delta_bin_0` a `x_delta_bin_9`: Distribución de movimiento horizontal
- `y_delta_bin_0` a `y_delta_bin_9`: Distribución de movimiento vertical
- `distance_bin_0` a `distance_bin_9`: Distribución de distancias
- `velocity_bin_0` a `velocity_bin_9`: Distribución de velocidades

### Interpretación de Clusters

**Cluster 0**: Trayectorias normales
- Movimiento regular y consistente
- Velocidad y distancia dentro de rangos esperados

**Cluster 1**: Trayectorias anómalas/sospechosas
- Representa ~0.02% de todas las trayectorias
- Características distintivas:
  - Total de puntos muy alto (promedio ~2.4M puntos)
  - Patrones de movimiento inusuales
  - Alta separación en espacio PCA

**Cluster 2**: Trayectorias de larga duración
- Trayectorias extensas pero con movimiento normal
- Promedio ~623K puntos

### Métricas de Calidad

En el análisis con k=3 se obtuvieron:
- **Silhouette Score**: 0.9389 (excelente separación)
- **Davies-Bouldin Index**: 0.8978 (buena compacidad)
- **Calinski-Harabasz Index**: 14901.70 (alta definición de clusters)

## Estructura del Proyecto

```
.
├── configs/                 # Archivos de configuración YAML
├── data/
│   ├── input_videos/       # Videos de entrada
│   ├── output_videos/      # Videos procesados
│   └── trajectories/       # Trayectorias exportadas (JSON)
├── detectors/              # Módulo de detección (YOLOv8)
├── trackers/               # Módulo de seguimiento (ByteTrack)
├── utils/                  # Utilidades (visualización, co-movimiento)
├── main.py                 # Script principal
├── export_trajectories.ipynb  # Exportar JSON → CSV
├── k_means.ipynb          # Entrenamiento de clustering
├── inference.ipynb        # Inferencia con modelo entrenado
└── requirements.txt       # Dependencias Python
```

## Notas Adicionales

- El sistema soporta procesamiento de múltiples videos en paralelo
- Las trayectorias se exportan incrementalmente cada N frames para optimizar memoria
- El agrupamiento de clases mejora la consistencia del tracking
- La detección de co-movimiento permite identificar asociaciones persona-objeto
- Los modelos entrenados son reutilizables para nuevos videos