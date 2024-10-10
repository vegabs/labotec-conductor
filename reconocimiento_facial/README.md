# Reconocimiento facial
## Entrenamiento
Para entrenar un nuevo modelo de reconocimiento de rostro se debe configurar archivo `training.py` modificando las siguientes variables:
- MODEL_FILENAME: nombre del modelo con extension ".clf"
- TEST_FOLDER_PATH = Ruta del folder donde se encuentras las imagenes de prueba. La estructura de este folder es la siguiente:

        <folder_prueba>/
        ├── <nombre_archivo1>
        ├── <nombre_archivo2>
        ├── <nombre_archivo3>
        └── ...

- TRAIN_FOLDER_PATH = Ruta donde se encuentran las imagenes de entrenamiento siguiento la siguiente estructura:

        <folder_entrenamiento>/
        ├── <nombre_persona1>/
        │   ├── <nombre_archivo1>.jpeg
        │   ├── <nombre_archivo2>.jpeg
        │   ├── ...
        ├── <nombre_persona2>/
        │   ├── <nombre_archivo1>.jpeg
        │   └── <nombre_archivo2>.jpeg
        └── ...

Los formatos de imagenes soportados son 'png', 'jpg', y 'jpeg'.

## Prueba
Para hacer la prueba ejecutar el archivo `test_in_video.py` y cambiar la variable `MODEL_FILENAME` con el nombre del modelo que se desea probar.