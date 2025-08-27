# Crear entorno virtual nuevo
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

# Instalar requirements:
```bash
pip install -r requirements.txt
```

# Instalar ByteTrack desde el repositorio oficial
```bash
git clone https://github.com/FoundationVision/ByteTrack.git byte_track_repo
cd /(raíz proyecto)/byte_track_repo
pip install -e .
```

## Verificar que se instaló correctamente
```bash
python -c "import yolox; print('YOLOX instalado correctamente.')"
```

# Comando para correr el programa
```bash
python main.py --video data/input_videos/{video_name}.mp4 --output data/output_videos/{video_name}.mp4 --config configs/{config_file}.yaml
```