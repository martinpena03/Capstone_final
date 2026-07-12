# -*- coding: utf-8 -*-
"""
camara_util.py — Módulo compartido para abrir la cámara de forma robusta.

Este archivo es un EJEMPLO DE MODULARIZACIÓN: en vez de copiar/pegar la
apertura de cámara en cada script (como pasa hoy en Ayolo_cuda.py y
Ayolo_jetson.py), la lógica vive en UN solo lugar y todos los scripts
la importan. Si mañana cambias de cámara, tocas solo este archivo.

CONCEPTOS CLAVE QUE HAY QUE ENTENDER DE OpenCV VideoCapture:

1) "Backend": OpenCV no habla directo con la cámara, usa un backend del
   sistema operativo:
     - Windows:  CAP_DSHOW (DirectShow, viejo pero confiable) o
                 CAP_MSMF (Media Foundation, moderno, a veces tarda en abrir).
     - Jetson/Linux: CAP_V4L2 (Video4Linux2) para cámaras USB.
     - Jetson con cámara CSI (la que se conecta por cinta): NO se usa
       VideoCapture(0), se usa un "pipeline" de GStreamer (ver más abajo).

2) "El buffer": la cámara produce frames a ritmo fijo (ej. 30 fps) aunque
   tu código no los lea. OpenCV guarda algunos en una cola interna.
   *** ESTE ES UNO DE LOS BUGS CENTRALES DE Ayolo_cuda.py ***:
   cuando el robot entra en "latch" deja de llamar cap.read() por hasta
   15 segundos; al volver a leer, recibe frames VIEJOS que estaban en la
   cola => el robot "ve el pasado" y cree que el obstáculo sigue/no sigue.
   Soluciones:
     a) cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  (no todos los backends lo respetan)
     b) NUNCA dejar de leer: un hilo dedicado lee siempre y se queda solo
        con el último frame (clase CamaraEnHilo de abajo).

3) set() MIENTE: pedir 1280x720 no garantiza recibir 1280x720. Siempre
   hay que verificar con get() o con frame.shape. Lo mismo con FOCUS,
   EXPOSURE, etc.: muchas cámaras ignoran la orden en silencio.

USO:
    from camara_util import abrir_camara, CamaraEnHilo, FuenteSintetica
    cap = abrir_camara(indice=0, ancho=1280, alto=720)
"""

import threading
import time

import cv2
import numpy as np


def abrir_camara(indice=0, ancho=1280, alto=720, fps=30, forzar_mjpg=True):
    """Abre una cámara probando los backends adecuados a cada sistema.

    Devuelve el objeto VideoCapture ya configurado, o None si no hay cámara.

    forzar_mjpg: las cámaras USB suelen ofrecer dos formatos:
      - YUYV (sin comprimir): a 1280x720 solo alcanza ~5-10 fps porque el
        cable USB no da para más datos crudos.
      - MJPG (comprimido): la cámara comprime internamente y sí llega a 30 fps.
      Pedir MJPG suele ser LA diferencia entre un sistema fluido y uno a pedales.
      En la Jetson esto se ve con:  v4l2-ctl --list-formats-ext
    """
    import platform

    if platform.system() == "Windows":
        # DSHOW abre más rápido y respeta más propiedades que MSMF.
        candidatos = [(indice, cv2.CAP_DSHOW), (indice, cv2.CAP_MSMF), (indice, cv2.CAP_ANY)]
    else:
        # Jetson / Linux con cámara USB.
        candidatos = [(indice, cv2.CAP_V4L2), (indice, cv2.CAP_ANY)]

    for idx, backend in candidatos:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if forzar_mjpg:
            # FOURCC = código de 4 letras del formato de pixel que le pedimos.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Cola mínima => siempre el frame más nuevo (si el backend lo respeta).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # "Warmup": el primer frame suele tardar (auto-exposición ajustándose).
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()

    return None


def pipeline_gstreamer_csi(ancho=1280, alto=720, fps=30, flip=0):
    """Pipeline de GStreamer para la cámara CSI de la Jetson (tipo Raspberry).

    SOLO aplica si la cámara va conectada por el cable plano (CSI). Para una
    webcam USB en la Jetson basta abrir_camara() con CAP_V4L2.

    Se usa así:
        cap = cv2.VideoCapture(pipeline_gstreamer_csi(), cv2.CAP_GSTREAMER)

    Requiere que el OpenCV de la Jetson esté compilado con GStreamer
    (el que trae JetPack sí lo está; el de `pip install opencv-python` NO).
    Verificar en la Jetson con:
        python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
    """
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={ancho}, height={alto}, "
        f"framerate={fps}/1 ! nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1"
        # drop=1 y max-buffers=1  ==> nunca acumular frames viejos.
    )


class CamaraEnHilo:
    """Lector de cámara en un hilo dedicado: SIEMPRE tienes el último frame.

    Esta es la solución correcta al problema del buffer: un hilo lee la
    cámara sin parar (y descarta lo viejo), y el resto del programa pide
    el frame más reciente cuando lo necesita, aunque se demore en procesar.

    Así la inferencia YOLO (lenta) nunca atrasa la captura, y cuando el
    planificador quiere "mirar de nuevo", lo que ve es AHORA y no hace 15 s.

    USO:
        cam = CamaraEnHilo(abrir_camara())
        cam.iniciar()
        ...
        ok, frame = cam.leer()   # nunca bloquea, frame más reciente
        ...
        cam.detener()
    """

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._corriendo = False
        self._hilo = None
        self.frames_leidos = 0          # para medir fps real de captura

    def iniciar(self):
        self._corriendo = True
        # daemon=True: si el programa principal muere, el hilo muere con él.
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        return self

    def _bucle(self):
        while self._corriendo:
            ok, frame = self.cap.read()     # bloquea hasta el próximo frame
            with self._lock:                # lock: nadie lee a medio escribir
                self._ok = ok
                if ok:
                    self._frame = frame
                    self.frames_leidos += 1

    def leer(self):
        """Devuelve (ok, copia_del_ultimo_frame). No bloquea."""
        with self._lock:
            if self._frame is None:
                return False, None
            # .copy() para que quien procese no pelee con el hilo que escribe.
            return self._ok, self._frame.copy()

    def detener(self):
        self._corriendo = False
        if self._hilo is not None:
            self._hilo.join(timeout=1.0)
        self.cap.release()


class FuenteSintetica:
    """Cámara falsa para trabajar SIN cámara (como ahora en Windows).

    Genera frames con una "pelota" y una "persona" de juguete moviéndose,
    con timestamp sobreimpreso (útil para VER el problema del buffer:
    si el timestamp del frame que procesas es viejo, estás leyendo cola).

    Imita la interfaz de VideoCapture (read/isOpened/release/get/set) para
    poder enchufarla donde iría la cámara real sin cambiar el resto del código
    — otro ejemplo de por qué las interfaces comunes (modularidad) sirven.
    """

    def __init__(self, ancho=1280, alto=720, fps=30):
        self.ancho, self.alto, self.fps = ancho, alto, fps
        self._t0 = time.time()
        self._ultimo = 0.0

    def isOpened(self):
        return True

    def read(self):
        # Respetar el ritmo de una cámara real (30 fps => 1 frame cada 33 ms).
        ahora = time.time()
        falta = (1.0 / self.fps) - (ahora - self._ultimo)
        if falta > 0:
            time.sleep(falta)
        self._ultimo = time.time()

        t = time.time() - self._t0
        frame = np.full((self.alto, self.ancho, 3), 40, np.uint8)

        # Piso cuadriculado de referencia (como el laberinto).
        for x in range(0, self.ancho, 80):
            cv2.line(frame, (x, 0), (x, self.alto), (60, 60, 60), 1)
        for y in range(0, self.alto, 80):
            cv2.line(frame, (0, y), (self.ancho, y), (60, 60, 60), 1)

        # "Pelota" que rebota de lado a lado.
        cx = int(self.ancho / 2 + (self.ancho / 3) * np.sin(t * 0.9))
        cy = int(self.alto * 0.72)
        cv2.circle(frame, (cx, cy), 55, (0, 140, 255), -1)
        cv2.circle(frame, (cx - 18, cy - 18), 12, (255, 255, 255), -1)

        # "Persona" de palitos que se acerca y aleja (cambia de tamaño).
        esc = 0.6 + 0.35 * np.sin(t * 0.35)
        px, alto_p = int(self.ancho * 0.25), int(260 * esc)
        base_y = int(self.alto * 0.86)
        cv2.line(frame, (px, base_y), (px, base_y - alto_p), (200, 200, 200), 6)
        cv2.circle(frame, (px, base_y - alto_p - 22), 22, (200, 200, 200), -1)

        cv2.putText(frame, f"SINTETICO  t={t:6.2f}s", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return True, frame

    def release(self):
        pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.ancho)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.alto)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop, valor):
        return False


def abrir_fuente(argv_tiene_sintetico=False, indice=0, ancho=1280, alto=720):
    """Atajo usado por los tutoriales: cámara real si hay, si no sintética."""
    if not argv_tiene_sintetico:
        cap = abrir_camara(indice=indice, ancho=ancho, alto=alto)
        if cap is not None:
            print("[camara_util] Camara real abierta.")
            return cap, True
        print("[camara_util] No se encontro camara. Usando fuente SINTETICA.")
    else:
        print("[camara_util] Fuente SINTETICA pedida por argumento.")
    return FuenteSintetica(ancho=ancho, alto=alto), False
