# -*- coding: utf-8 -*-
"""
camara_y_deteccion.py — Vision del robot de bodega.

Responsabilidad (PRD seccion A): procesar el feed de la camara, detectar
obstaculos con YOLO, coordinar con laberinto_y_a_star.py el flujo de
detencion / re-evaluacion / recalculo, y transmitir el video (POV) al
dashboard web.

Flujo de obstaculos (PRD A.1-A.3):
    0. Un obstaculo detectado se atribuye al NODO AL FRENTE del nodo actual
       (la celda contigua hacia la que mira el robot). Si entre ambas celdas
       hay un muro, la deteccion se descarta: no se puede ver un obstaculo
       a traves de una pared.
    1. Deteccion confirmada (N frames seguidos) -> flag_obs=1. El planificador
       responde publicando refs (-10,-10): el robot se detiene.
    2. Mientras flag_obs este activa se re-verifica cada --intervalo-reevaluacion
       segundos CON FRAMES FRESCOS (la captura corre en un hilo dedicado y
       nunca se detiene, asi jamas se procesan frames viejos del buffer).
    3. Si el obstaculo se retira antes de --timeout-bloqueo -> flag_obs=0 y el
       robot retoma su camino con las referencias previas.
    4. Si persiste mas alla del timeout -> se publica el nodo bloqueado, el
       planificador lo marca en su matriz, recalcula A*, y se limpia flag_obs
       para que el robot siga por el desvio.

Todos los tiempos y umbrales son PARAMETROS de linea de comandos (PRD A.2/A.3):
nada esta hardcodeado.

Ejecutar (Jetson):   python 02_camara_y_deteccion.py
Ejecutar (PC, sin camara):  python 02_camara_y_deteccion.py --sintetico
Smoke test:          python 02_camara_y_deteccion.py --sintetico --test
Ensayo determinista del ciclo completo de obstaculo (sin YOLO):
    python 02_camara_y_deteccion.py --sintetico --test-obstaculo 12,5,60
    (obstaculo falso en el nodo 12, aparece a los 5 s, dura 60 s)
    Con NODO="frente" el obstaculo falso se ubica en el nodo al frente del
    robot, ejercitando la misma logica de atribucion que YOLO.
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from topicos import mqtt_topics
from camara_util import abrir_fuente, CamaraEnHilo
from yolo_backend import cargar_detector

# Fallback del grafo del laberinto (espejo del grafo base del planificador).
# Solo se usa hasta que llegue el grafo retenido por MQTT, que es la fuente
# de verdad en runtime (incluye las ediciones hechas desde la web).
CONEXIONES_BASE = {
    "00": ["10", "01"], "01": ["00", "02"], "02": ["01"], "03": ["04", "13"], "04": ["03", "14"],
    "10": ["00", "20"], "11": ["12", "21"], "12": ["11", "13", "22"], "13": ["03", "12"], "14": ["04", "24"],
    "20": ["10", "30"], "21": ["22", "11"], "22": ["21", "32", "12", "23"], "23": ["24", "33", "22"], "24": ["14", "23", "34"],
    "30": ["31", "20"], "31": ["30", "32"], "32": ["22", "31"], "33": ["23", "43"], "34": ["24"],
    "40": ["50", "41"], "41": ["40", "42"], "42": ["41"], "43": ["33", "44"], "44": ["43", "54"],
    "50": ["40", "51"], "51": ["50", "52"], "52": ["51", "53"], "53": ["52", "54"], "54": ["53", "44"],
}

# Clases COCO consideradas obstaculo, POR NOMBRE. Los ids se resuelven contra
# model.names al cargar el detector (los ids escritos a mano del sistema viejo
# estaban mal rotulados: 67 es "cell phone", no "mesa").
CLASES_OBSTACULO_DEFECTO = (
    "person", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
    "bottle", "cup", "chair", "couch", "potted plant", "dining table", "book",
)


# ============================================================================
# GEOMETRIA: pixel -> distancia -> posicion global -> nodo
# ============================================================================
def cargar_intrinsecos(directorio):
    """Carga la calibracion REAL de la camara (camera_matrix.npy). El sistema
    viejo usaba valores inventados (~25% de error medido); si los .npy no
    estan, se cae a esos valores pero avisando fuerte."""
    ruta = os.path.join(directorio, "camera_matrix.npy")
    if os.path.exists(ruta):
        K = np.load(ruta)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        print(f"[CAM] Calibracion real: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        return fx, fy, cx, cy
    print("[CAM][AVISO] Sin camera_matrix.npy: usando parametros nominales "
          "(la distancia tendra error; calibrar en el robot).")
    return 910.0, 910.0, 640.0, 360.0


def distancia_horizontal(y_base, cy, fy, altura_cm, pitch_deg):
    """Distancia sobre el piso a un objeto cuya base toca la fila y_base.
    Modelo: camara a altura H con inclinacion pitch hacia el piso.
    D = H / tan(pitch + alpha). None si el rayo no corta el piso adelante."""
    alpha = math.atan2(y_base - cy, fy)
    angulo = math.radians(pitch_deg) + alpha
    if math.tan(angulo) <= 0:
        return None
    return altura_cm / math.tan(angulo)


class DetectorObstaculos:
    """Maquina de estados de obstaculos + streaming de video hacia la web."""

    ESTADO_LIBRE = "LIBRE"
    ESTADO_OBSTACULO = "OBSTACULO"

    def __init__(self, args):
        self.args = args
        dir_script = os.path.dirname(os.path.abspath(__file__))
        self.fx, self.fy, self.cx, self.cy = cargar_intrinsecos(dir_script)

        # Rumbo del robot EN EL MARCO DEL LABERINTO (grados, antihorario
        # desde el Este) y su nodo actual, ambos publicados por
        # laberinto_y_a_star (que es quien conoce la pose inicial declarada
        # en la web). Con eso se determina el NODO AL FRENTE.
        self.robot_theta = 0.0
        self.nodo_actual = None

        # Conexiones del laberinto: llegan con el grafo retenido del
        # planificador (unica fuente de verdad, incluye ediciones de la web);
        # fallback a la copia local. Sirven para saber si hay un MURO entre
        # el nodo actual y el nodo al frente.
        self.graph_connections = {n: list(v) for n, v in CONEXIONES_BASE.items()}

        # Nodos ya bloqueados por el planificador (vienen en el payload del
        # grafo retenido). Un obstaculo sobre un nodo bloqueado ya esta
        # resuelto: NO debe volver a levantar flag_obs, o el robot se
        # detendria una y otra vez frente a un bloqueo que A* ya rodeo.
        self.nodos_bloqueados = set()

        # Estado del ciclo de obstaculo
        self.estado = self.ESTADO_LIBRE
        self.racha_presente = 0
        self.racha_ausente = 0
        self.nodo_detectado = None
        self.t_deteccion = None
        self.t_proxima_reevaluacion = None
        self.ultima_distancia = None

        self.client = None
        self.detector = None
        self.clases_ids = None

    # ------------------------------------------------------------------
    # MQTT
    # ------------------------------------------------------------------
    def conectar(self):
        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.args.broker, self.args.puerto, 60)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Conectado al broker {self.args.broker}:{self.args.puerto}")
        # Rumbo y nodo actual del robot en el marco del laberinto
        # (desde laberinto_y_a_star)
        client.subscribe(mqtt_topics["camara"]["theta_cam"])
        client.subscribe(mqtt_topics["camara"]["nodo_actual"])
        # Estructura del laberinto (retenida): conexiones y nodos bloqueados
        client.subscribe(mqtt_topics["planificador"]["grafo"])

    def _on_message(self, client, userdata, msg):
        try:
            if msg.topic == mqtt_topics["planificador"]["grafo"]:
                import json
                payload = json.loads(msg.payload.decode())
                conexiones = payload.get("graph_connections")
                if isinstance(conexiones, dict) and conexiones:
                    self.graph_connections = conexiones
                self.nodos_bloqueados = set(payload.get("nodos_bloqueados") or [])
                return

            if msg.topic == mqtt_topics["camara"]["nodo_actual"]:
                nodo = msg.payload.decode().strip()
                if len(nodo) == 2 and nodo.isdigit():
                    self.nodo_actual = nodo
                return

            if msg.topic == mqtt_topics["camara"]["theta_cam"]:
                self.robot_theta = float(msg.payload.decode().strip())
        except Exception as e:
            print(f"[MQTT ERROR] {msg.topic}: {e}")

    def _publicar(self, topico, payload, retain=False):
        if self.client:
            self.client.publish(topico, payload, retain=retain)

    # ------------------------------------------------------------------
    # DETECCION
    # ------------------------------------------------------------------
    def cargar_yolo(self):
        self.detector = cargar_detector(conf=self.args.conf,
                                        imgsz=self.args.imgsz,
                                        preferir=self.args.motor)
        # Resolver los NOMBRES de clase a ids contra el modelo cargado.
        nombres = self.detector.names
        if isinstance(nombres, dict):
            nombre_a_id = {v: k for k, v in nombres.items()}
        else:
            nombre_a_id = {v: k for k, v in enumerate(nombres)}
        pedidas = [c.strip() for c in self.args.clases.split(",") if c.strip()]
        self.clases_ids = set()
        for nombre in pedidas:
            if nombre in nombre_a_id:
                self.clases_ids.add(nombre_a_id[nombre])
            else:
                print(f"[YOLO][AVISO] Clase '{nombre}' no existe en el modelo; ignorada.")
        activas = sorted(nombres[i] for i in self.clases_ids)
        print(f"[YOLO] Clases obstaculo activas: {activas}")

    def nodo_al_frente(self):
        """Nodo contiguo hacia el que mira el robot, o None si no se le puede
        atribuir un obstaculo.

        La regla del sistema: un obstaculo visto por la camara SIEMPRE se
        asume en la celda contigua al frente del nodo actual (el robot mira
        por el pasillo que va a recorrer). Devuelve None cuando:
          - aun no se conoce el nodo actual (el planificador no ha publicado),
          - el frente queda fuera de la grilla, o
          - hay un MURO entre ambas celdas: no tiene sentido "ver" un
            obstaculo a traves de una pared, asi que la deteccion se descarta.
        """
        actual = self.nodo_actual
        if not actual:
            return None

        # Cuantizar el rumbo (grados antihorario desde el Este) al eje
        # cardinal mas cercano. Labels: primer digito columna, segundo fila.
        theta = self.robot_theta % 360.0
        if theta >= 315.0 or theta < 45.0:
            d_col, d_fila = 1, 0     # Este
        elif theta < 135.0:
            d_col, d_fila = 0, 1     # Norte
        elif theta < 225.0:
            d_col, d_fila = -1, 0    # Oeste
        else:
            d_col, d_fila = 0, -1    # Sur

        col, fila = int(actual[0]) + d_col, int(actual[1]) + d_fila
        if not (0 <= col <= 9 and 0 <= fila <= 9):
            return None
        frente = f"{col}{fila}"
        if frente not in self.graph_connections:
            return None

        # Conexion mutua = pasillo abierto. Si falta, hay un muro entre medio.
        if (frente not in self.graph_connections.get(actual, [])
                or actual not in self.graph_connections.get(frente, [])):
            return None
        return frente

    def evaluar_frame(self, frame):
        """Corre YOLO sobre el frame y devuelve (hay_obstaculo, nodo, dist_cm,
        detecciones_dibujables). Una deteccion cuenta como obstaculo si su
        clase esta en la lista y su base cae en el rango de distancias util;
        el nodo afectado es SIEMPRE el nodo al frente del robot (ver
        nodo_al_frente): si el robot mira un muro o fuera de la grilla, la
        deteccion se descarta."""
        detecciones = self.detector.detectar(frame)
        menor_dist = None
        for det in detecciones:
            if det.clase_id not in self.clases_ids:
                continue
            _, y_base = det.base
            dist = distancia_horizontal(y_base, self.cy, self.fy,
                                        self.args.altura_cam, self.args.pitch_cam)
            if dist is None or not (self.args.dist_min <= dist <= self.args.dist_max):
                continue
            if menor_dist is None or dist < menor_dist:
                menor_dist = dist

        if menor_dist is None:
            return False, None, None, detecciones

        nodo = self.nodo_al_frente()
        if nodo is None:
            return False, None, None, detecciones
        return True, nodo, menor_dist, detecciones

    # ------------------------------------------------------------------
    # MAQUINA DE ESTADOS (PRD A.1-A.3)
    # ------------------------------------------------------------------
    def procesar(self, hay_obstaculo, nodo, dist):
        ahora = time.time()

        # Un obstaculo mapeado a un nodo YA bloqueado no es novedad: el
        # planificador ya lo rodeo. Se ignora para no re-detener al robot.
        if hay_obstaculo and nodo in self.nodos_bloqueados:
            hay_obstaculo = False
            nodo = None
            dist = None

        if self.estado == self.ESTADO_LIBRE:
            self.racha_presente = self.racha_presente + 1 if hay_obstaculo else 0
            # Persistencia temporal: N frames seguidos evitan que un unico
            # falso positivo de YOLO congele el sistema (bug del sistema viejo).
            if self.racha_presente >= self.args.n_persistencia:
                self.estado = self.ESTADO_OBSTACULO
                self.nodo_detectado = nodo
                self.ultima_distancia = dist
                self.t_deteccion = ahora
                self.t_proxima_reevaluacion = ahora + self.args.intervalo_reevaluacion
                self.racha_ausente = 0
                # flag_obs: el planificador respondera con refs (-10,-10),
                # la señal de detencion que la ESP ya entiende (PRD A.1).
                # === ENLACE WEB (MQTT/WebSocket) — aviso de obstáculo activo (flag_obs) al planificador y al dashboard ===
                self._publicar(mqtt_topics["estados"]["flag_obs"], "1")
                # === ENLACE WEB (MQTT/WebSocket) — envío del nodo y distancia del obstáculo al dashboard (cruz parpadeante) ===
                self._publicar(mqtt_topics["camara"]["nodo_obs"], str(nodo))
                self._publicar(mqtt_topics["camara"]["dist_obs"], f"{dist:.1f}")
                print(f"[OBSTACULO] Confirmado en nodo {nodo} a {dist:.1f} cm. "
                      f"flag_obs=1 (robot detenido).")
            return

        # estado OBSTACULO: seguimos mirando SIEMPRE (frames frescos del hilo)
        self.racha_ausente = 0 if hay_obstaculo else self.racha_ausente + 1
        if hay_obstaculo:
            self.ultima_distancia = dist

        if ahora < self.t_proxima_reevaluacion:
            return
        self.t_proxima_reevaluacion = ahora + self.args.intervalo_reevaluacion

        # Re-evaluacion periodica (PRD A.2)
        if self.racha_ausente >= self.args.n_persistencia:
            print(f"[OBSTACULO] El nodo {self.nodo_detectado} quedo libre antes del "
                  f"timeout. flag_obs=0 (el robot retoma su camino).")
            self._limpiar(bloqueado=False)
            return

        # Timeout maximo (PRD A.3): el obstaculo es permanente -> bloquear nodo
        if ahora - self.t_deteccion >= self.args.timeout_bloqueo:
            print(f"[OBSTACULO] Timeout de {self.args.timeout_bloqueo:.0f} s superado: "
                  f"bloqueando nodo {self.nodo_detectado} y recalculando ruta.")
            # Primero el bloqueo (el planificador quita el nodo y recalcula)...
            self._publicar(mqtt_topics["camara"]["nodo_bloqueado"],
                           str(self.nodo_detectado))
            time.sleep(0.2)  # dar tiempo a que el bloqueo llegue antes que el go
            # ...y despues se libera el flag para que el robot siga por el desvio.
            self._limpiar(bloqueado=True)

    def _limpiar(self, bloqueado):
        self._publicar(mqtt_topics["estados"]["flag_obs"], "0")
        if not bloqueado:
            self._publicar(mqtt_topics["camara"]["nodo_obs"], "ninguno")
            self._publicar(mqtt_topics["camara"]["dist_obs"], "-1")
        self.estado = self.ESTADO_LIBRE
        self.racha_presente = 0
        self.racha_ausente = 0
        self.nodo_detectado = None
        self.t_deteccion = None

    # ------------------------------------------------------------------
    # STREAMING DE VIDEO HACIA EL DASHBOARD (PRD A.4)
    # ------------------------------------------------------------------
    def publicar_video(self, frame, detecciones):
        """Publica el frame como JPEG binario. Reglas sanas para no castigar
        al broker que tambien mueve las referencias del robot: <=10 fps,
        <=640 px de ancho, calidad ~60 (todo configurable por parametro)."""
        for det in detecciones or []:
            cv2.rectangle(frame, (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)), (0, 220, 0), 2)
            cv2.putText(frame, f"{det.nombre} {det.conf:.2f}",
                        (int(det.x1), max(18, int(det.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        if self.estado == self.ESTADO_OBSTACULO:
            cv2.putText(frame, f"OBSTACULO nodo {self.nodo_detectado}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        alto, ancho = frame.shape[:2]
        if ancho > self.args.ancho_video:
            frame = cv2.resize(frame, (self.args.ancho_video,
                                       int(alto * self.args.ancho_video / ancho)))
        ok, jpeg = cv2.imencode(".jpg", frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.args.calidad_video])
        if ok and self.client:
            # === ENLACE WEB (MQTT/WebSocket) — streaming de video POV del robot al dashboard ===
            # qos=0 y retain=False: un frame perdido no importa, ya viene otro.
            self.client.publish(mqtt_topics["camara"]["video"],
                                jpeg.tobytes(), qos=0, retain=False)


# ============================================================================
# OBSTACULO SINTETICO PARA ENSAYOS (--test-obstaculo NODO,INICIO,DURACION)
# ============================================================================
class ObstaculoDePrueba:
    """Inyecta una 'deteccion' falsa en un nodo dado durante una ventana de
    tiempo, sin YOLO de por medio. Permite ensayar el ciclo completo
    flag_obs -> re-evaluacion -> timeout -> bloqueo de forma determinista."""

    def __init__(self, espec):
        try:
            nodo, inicio, duracion = espec.split(",")
            self.nodo = nodo.strip()
            self.t_inicio = float(inicio)
            self.t_fin = self.t_inicio + float(duracion)
        except ValueError:
            raise SystemExit("--test-obstaculo espera NODO,INICIO_S,DURACION_S "
                             "(ej: 12,5,60)")
        self.t0 = time.time()
        print(f"[TEST] Obstaculo sintetico en nodo {self.nodo}: aparece a los "
              f"{self.t_inicio:.0f} s y dura {self.t_fin - self.t_inicio:.0f} s.")

    def activo(self):
        t = time.time() - self.t0
        return self.t_inicio <= t < self.t_fin


def main():
    parser = argparse.ArgumentParser(
        description="Vision del robot: deteccion de obstaculos + streaming POV")
    parser.add_argument("--broker", default="localhost", help="host del broker MQTT")
    parser.add_argument("--puerto", type=int, default=1883, help="puerto del broker")
    parser.add_argument("--camara", type=int, default=0, help="indice de la camara")
    parser.add_argument("--sintetico", action="store_true",
                        help="usar fuente de video sintetica (sin camara fisica)")
    # Parametros de deteccion (PRD: configurables, no hardcodeados)
    parser.add_argument("--conf", type=float, default=0.40,
                        help="umbral de confianza de YOLO")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="tamano de entrada de YOLO (416 acelera en Jetson)")
    parser.add_argument("--motor", default="auto",
                        choices=("auto", "ultralytics", "yolov5"),
                        help="motor YOLO: ultralytics (PC) o yolov5 torch.hub (Jetson)")
    parser.add_argument("--clases", default=",".join(CLASES_OBSTACULO_DEFECTO),
                        help="clases COCO consideradas obstaculo (por nombre)")
    parser.add_argument("--n-persistencia", type=int, default=3, dest="n_persistencia",
                        help="frames consecutivos para confirmar (o descartar) un obstaculo")
    parser.add_argument("--intervalo-reevaluacion", type=float, default=1.0,
                        dest="intervalo_reevaluacion",
                        help="segundos entre re-verificaciones del obstaculo (PRD A.2)")
    parser.add_argument("--timeout-bloqueo", type=float, default=10.0,
                        dest="timeout_bloqueo",
                        help="segundos de espera maxima antes de bloquear el nodo (PRD A.3)")
    parser.add_argument("--dist-min", type=float, default=7.0, dest="dist_min",
                        help="cm minimos para considerar un obstaculo")
    parser.add_argument("--dist-max", type=float, default=120.0, dest="dist_max",
                        help="cm maximos para considerar un obstaculo")
    # Geometria de la camara montada (medir en el robot; ver tutorial 04)
    parser.add_argument("--altura-cam", type=float, default=15.0, dest="altura_cam",
                        help="altura del lente al piso, cm")
    parser.add_argument("--pitch-cam", type=float, default=20.0, dest="pitch_cam",
                        help="inclinacion de la camara hacia el piso, grados")
    # Streaming de video (PRD A.4)
    parser.add_argument("--fps-video", type=float, default=8.0, dest="fps_video",
                        help="fps del streaming hacia la web")
    parser.add_argument("--ancho-video", type=int, default=480, dest="ancho_video",
                        help="ancho en px del video transmitido")
    parser.add_argument("--calidad-video", type=int, default=60, dest="calidad_video",
                        help="calidad JPEG (1-100) del video transmitido")
    # Modos de prueba
    parser.add_argument("--test", action="store_true",
                        help="smoke test: 15 frames sin broker y salir")
    parser.add_argument("--test-obstaculo", default=None, dest="test_obstaculo",
                        help="NODO,INICIO_S,DURACION_S — inyecta un obstaculo falso "
                             "(NODO='frente' lo ubica en el nodo al frente del robot)")
    args = parser.parse_args()

    det = DetectorObstaculos(args)

    if args.test:
        # Auto-verificacion de la atribucion al NODO AL FRENTE (grafo base)
        casos = [
            ("21", 0.0, None, "mirando un muro (21->31) se descarta"),
            ("21", 90.0, "22", "pasillo abierto al Norte (21->22)"),
            ("20", 0.0, "30", "pasillo abierto al Este (20->30)"),
            ("00", 180.0, None, "mirando fuera de la grilla se descarta"),
            (None, 0.0, None, "sin nodo actual aun, se descarta"),
        ]
        for nodo_act, theta, esperado, descripcion in casos:
            det.nodo_actual, det.robot_theta = nodo_act, theta
            resultado = det.nodo_al_frente()
            estado = "OK" if resultado == esperado else "FALLA"
            print(f"  [{estado}] nodo_al_frente({nodo_act}, {theta:.0f} deg) = "
                  f"{resultado} — {descripcion}")
        det.nodo_actual, det.robot_theta = None, 0.0

    obstaculo_fake = ObstaculoDePrueba(args.test_obstaculo) if args.test_obstaculo else None
    usar_yolo = obstaculo_fake is None
    if usar_yolo:
        det.cargar_yolo()

    if not args.test:
        det.conectar()

    cap, _ = abrir_fuente(argv_tiene_sintetico=args.sintetico, indice=args.camara)
    cam = CamaraEnHilo(cap).iniciar()
    print("[INFO] camara_y_deteccion corriendo. Ctrl+C para salir.")

    periodo_video = 1.0 / args.fps_video if args.fps_video > 0 else None
    t_ultimo_video = 0.0
    frames_procesados = 0

    try:
        while True:
            ok, frame = cam.leer()
            if not ok:
                time.sleep(0.05)
                continue

            # 1) Evaluacion del frame (YOLO real o obstaculo de ensayo)
            if usar_yolo:
                hay, nodo, dist, detecciones = det.evaluar_frame(frame)
            else:
                hay = obstaculo_fake.activo()
                if hay and obstaculo_fake.nodo == "frente":
                    # Misma atribucion que YOLO: nodo contiguo al frente
                    nodo = det.nodo_al_frente()
                    hay = nodo is not None
                else:
                    nodo = obstaculo_fake.nodo if hay else None
                dist = 50.0 if hay else None
                detecciones = []

            # 2) Maquina de estados de obstaculos
            det.procesar(hay, nodo, dist)

            # 3) Streaming POV hacia el dashboard, decimado a fps_video
            ahora = time.time()
            if periodo_video and (ahora - t_ultimo_video) >= periodo_video:
                det.publicar_video(frame, detecciones)
                t_ultimo_video = ahora

            frames_procesados += 1
            if args.test and frames_procesados >= 15:
                print("[TEST OK] 15 frames procesados sin errores.")
                break

            # Ritmo del lazo de vision: la captura vive en su hilo, asi que
            # este sleep NO acumula frames viejos (siempre se lee el ultimo).
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[INFO] Finalizado por teclado.")
    finally:
        # Estado seguro: nunca dejar el robot detenido por un flag huerfano.
        if det.client:
            det._publicar(mqtt_topics["estados"]["flag_obs"], "0")
            time.sleep(0.2)
            det.client.loop_stop()
            det.client.disconnect()
        cam.detener()


if __name__ == "__main__":
    main()
