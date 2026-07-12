# -*- coding: utf-8 -*-
"""
yolo_backend.py — Una sola interfaz para dos mundos: PC y Jetson.

PROBLEMA REAL DEL PROYECTO: en el PC conviene usar el paquete moderno
`ultralytics` (YOLOv8/v11, API simple), pero la Jetson Nano corre Python
viejo y ahí el código existente usa `torch.hub` con YOLOv5. Si cada script
llama directo a una de las dos APIs, nada es portable.

SOLUCIÓN (y lección de modularización): definimos NUESTRA interfaz mínima
—`detectar(frame) -> lista de Deteccion`— y escondemos detrás cuál motor
la implementa. Los tutoriales y el futuro código del robot hablan solo
con esta interfaz. Cambiar de motor = cambiar una línea.

CONCEPTOS DE YOLO QUE APARECEN AQUÍ (los ajustas en vivo en 03_yolo_basico):

  conf (confianza, 0-1): umbral para aceptar una detección. Bajo (0.25):
       detecta más, con más falsos positivos. Alto (0.6): solo lo seguro.
       El código actual usa 0.40 y aun así hay falsos positivos porque
       basta UN frame malo para gatillar todo (falta persistencia temporal,
       eso se enseña en 04 y se propone en el plan).

  iou (0-1): umbral del NMS (Non-Max Suppression). YOLO propone muchas
       cajas superpuestas para el mismo objeto; el NMS se queda con la
       mejor y elimina las que se superponen más que `iou`.

  imgsz: tamaño al que se reescala la imagen antes de entrar a la red
       (siempre cuadrado, típicamente 640). MENOS tamaño = MÁS rápido y
       MENOS precisión con objetos chicos. En la Nano, 416 o 320 puede
       ser la diferencia entre 4 fps y 10 fps.

  classes: filtrar por clase EN el modelo es más eficiente y limpio que
       filtrar después a mano (como hace el código actual con
       CLASES_OBSTACULO).

  half (FP16): en GPU usa números de 16 bits => ~2x más rápido en Jetson.
       En CPU no aplica.

OJO CON LOS IDs DE CLASE (error encontrado en el código actual):
  El dataset COCO tiene 80 clases con índices fijos. En Ayolo_cuda.py el
  diccionario dice {26: "paraguas", 28: "bolso", 67: "mesa"} pero en COCO:
     25 = umbrella (paraguas)   26 = handbag (cartera)
     28 = suitcase (maleta)     60 = dining table (mesa)
     67 = cell phone (¡CELULAR, no mesa!)
  O sea: el sistema actual trata los CELULARES como mesas y nunca vio una
  mesa como obstáculo. Moraleja: los nombres SIEMPRE se leen del modelo
  (model.names), jamás se escriben a mano.
"""

import os


class Deteccion:
    """Una caja detectada. Coordenadas en píxeles de la imagen ORIGINAL."""

    __slots__ = ("x1", "y1", "x2", "y2", "conf", "clase_id", "nombre")

    def __init__(self, x1, y1, x2, y2, conf, clase_id, nombre):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.conf = float(conf)
        self.clase_id = int(clase_id)
        self.nombre = str(nombre)

    @property
    def base(self):
        """Punto medio del borde INFERIOR de la caja: donde el objeto toca el
        piso. Es el píxel que se usa para estimar distancia (ver tutorial 04)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def __repr__(self):
        return (f"{self.nombre}({self.conf:.2f}) "
                f"[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}]")


class DetectorUltralytics:
    """Motor moderno (PC): paquete `ultralytics`, modelos YOLOv8/v11.

    El primer uso descarga el modelo (~6 MB para yolov8n). 'n' = nano,
    el más chico y rápido; s/m/l/x suben precisión y costo.
    """

    def __init__(self, modelo="yolov8n.pt", conf=0.4, iou=0.45, imgsz=640, clases=None):
        from ultralytics import YOLO   # import adentro: solo si se usa este motor
        self.model = YOLO(modelo)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.clases = list(clases) if clases else None
        self.names = self.model.names          # dict {id: nombre} DEL MODELO

    def detectar(self, frame_bgr):
        # verbose=False: sin spam por consola en cada frame.
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, classes=self.clases,
                                 verbose=False)[0]
        salida = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            salida.append(Deteccion(x1, y1, x2, y2, float(b.conf[0]), cid,
                                    self.names.get(cid, f"clase_{cid}")))
        return salida


class DetectorYolov5Hub:
    """Motor de la Jetson: torch.hub + YOLOv5 (lo mismo que usa Ayolo_cuda.py).

    Reproduce EXACTAMENTE el camino del código actual para que lo que
    aprendas aquí aplique 1:1 en la Jetson:
      - carga desde el repo local ./yolov5 si existe (sin internet), o
        desde GitHub (ultralytics/yolov5) si no,
      - model.conf / model.iou / model.classes son atributos del modelo,
      - en CUDA se puede usar model.half() para FP16.
    """

    def __init__(self, pesos="yolov5s.pt", conf=0.4, iou=0.45, imgsz=640,
                 clases=None, repo_local=None):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if repo_local and os.path.isdir(repo_local):
            self.model = torch.hub.load(repo_local, "custom", path=pesos, source="local")
        else:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=pesos)

        self.model.to(self.device)
        self.model.conf = conf
        self.model.iou = iou
        self.model.classes = list(clases) if clases else None   # filtro nativo
        self.model.eval()
        if self.device.type == "cuda":
            self.model.half()    # FP16: ~2x en Jetson, imperceptible en precisión
        self.imgsz = imgsz
        self.names = self.model.names

    def detectar(self, frame_bgr):
        import cv2
        # YOLOv5-hub espera RGB; OpenCV entrega BGR. Si te lo saltas, los
        # colores quedan invertidos y la precisión baja unos puntos.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():           # sin gradientes: inferencia pura
            res = self.model(rgb, size=self.imgsz)
        salida = []
        pred = res.xyxy[0]
        if pred is not None:
            for x1, y1, x2, y2, conf, cid in pred.cpu().numpy():
                cid = int(cid)
                nombre = self.names[cid] if cid < len(self.names) else f"clase_{cid}"
                salida.append(Deteccion(x1, y1, x2, y2, conf, cid, nombre))
        return salida


def cargar_detector(conf=0.4, iou=0.45, imgsz=640, clases=None, preferir="auto"):
    """Fábrica: devuelve el detector disponible en esta máquina.

    preferir: "auto" (ultralytics si está instalado, si no yolov5-hub),
              "ultralytics" o "yolov5".
    """
    if preferir in ("auto", "ultralytics"):
        try:
            det = DetectorUltralytics(conf=conf, iou=iou, imgsz=imgsz, clases=clases)
            print("[yolo_backend] Motor: ultralytics (YOLOv8n).")
            return det
        except ImportError:
            if preferir == "ultralytics":
                raise
            print("[yolo_backend] ultralytics no instalado, probando torch.hub YOLOv5...")

    dir_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det = DetectorYolov5Hub(
        pesos=os.path.join(dir_proyecto, "yolov5s.pt"),
        repo_local=os.path.join(dir_proyecto, "yolov5"),
        conf=conf, iou=iou, imgsz=imgsz, clases=clases,
    )
    print("[yolo_backend] Motor: torch.hub YOLOv5 (igual que la Jetson).")
    return det
