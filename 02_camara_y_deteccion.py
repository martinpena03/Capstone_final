# -*- coding: utf-8 -*-
"""
camara_y_deteccion.py — Vision del robot de bodega.

Responsabilidad (PRD seccion A): procesar el feed de la camara, detectar
obstaculos con YOLO, coordinar con laberinto_y_a_star.py el flujo de
detencion / re-evaluacion / recalculo, y transmitir el video (POV) al
dashboard web.

Localizacion absoluta por marcadores ArUco (reemplaza al prototipo AAY.py):
    * Sobre el MISMO frame y con la MISMA calibracion intrinseca que la
      deteccion de obstaculos, se detectan marcadores DICT_4X4_50 y se estima
      la pose absoluta del robot en el marco del laberinto (solvePnP).
    * Cada marcador se ancla a una CELDA + PARED (N/S/E/W) con la cara
      mirando hacia adentro de esa celda; la asignacion se edita desde la web
      y PERSISTE en marcadores.json hasta la proxima edicion.
    * Si el editor del laberinto elimina una pared, los marcadores pegados
      en sus caras se borran automaticamente.
    * La pose calculada (corregida por el offset camara->centro del robot)
      se publica en robot/camara/pose_absoluta para que el planificador
      SOBREESCRIBA su odometria. Un toggle maestro desde la web la apaga.

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
import json
import math
import os
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from topicos import mqtt_topics
from camara_util import abrir_fuente, CamaraEnHilo
from yolo_backend import cargar_detector

# ============================================================================
# CONSTANTES DE LA LOCALIZACION POR MARCADORES ARUCO (editar aqui)
# ============================================================================
MARKER_SIZE_CM = 4.0       # lado del cuadrado negro del ArUco impreso, en cm
OFFSET_CAMARA_CM = 10.0    # el centro de giro del robot esta 10 cm DETRAS de la lente
CELL_SIZE = 30             # cm entre centros de celdas contiguas (igual que el planificador)
ARCHIVO_MARCADORES = "marcadores.json"  # persistencia de la asignacion (junto al script)

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

    "bottle", "cup",  "potted plant",
)


# ============================================================================
# GEOMETRIA: pixel -> distancia -> posicion global -> nodo
# ============================================================================
def cargar_intrinsecos(directorio):
    """Carga la calibracion REAL de la camara (camera_matrix.npy y
    dist_coeffs.npy). Es UNA SOLA calibracion para todo el modulo: la
    geometria de obstaculos usa fy/cy de la matriz K, y el PnP de los
    marcadores ArUco usa K + coeficientes de distorsion. El sistema viejo
    usaba valores inventados (~25% de error medido); si los .npy no estan,
    se cae a esos valores pero avisando fuerte."""
    ruta_k = os.path.join(directorio, "camera_matrix.npy")
    ruta_d = os.path.join(directorio, "dist_coeffs.npy")
    if os.path.exists(ruta_k):
        K = np.load(ruta_k).astype(np.float64)
        if os.path.exists(ruta_d):
            dist = np.load(ruta_d).astype(np.float64)
        else:
            print("[CAM][AVISO] Sin dist_coeffs.npy: PnP sin corregir distorsion.")
            dist = np.zeros(5)
        print(f"[CAM] Calibracion real: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} "
              f"cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
        return K, dist
    print("[CAM][AVISO] Sin camera_matrix.npy: usando parametros nominales "
          "(la distancia tendra error; calibrar en el robot).")
    K = np.array([[910.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]])
    return K, np.zeros(5)


def distancia_horizontal(y_base, cy, fy, altura_cm, pitch_deg):
    """Distancia sobre el piso a un objeto cuya base toca la fila y_base.
    Modelo: camara a altura H con inclinacion pitch hacia el piso.
    D = H / tan(pitch + alpha). None si el rayo no corta el piso adelante."""
    alpha = math.atan2(y_base - cy, fy)
    angulo = math.radians(pitch_deg) + alpha
    if math.tan(angulo) <= 0:
        return None
    return altura_cm / math.tan(angulo)


# ============================================================================
# LOCALIZACION ABSOLUTA POR MARCADORES ARUCO
# ============================================================================
# Delta (columna, fila) hacia la celda vecina que queda del otro lado de la
# pared, por cada cara. La grilla crece: X (columna) al Este, Y (fila) al Norte.
PAREDES_VECINO = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
# Angulo (grados, antihorario desde el Este) de la NORMAL SALIENTE del
# marcador: desde la pared hacia adentro de la celda que lo declara.
PAREDES_NORMAL = {"N": 270.0, "S": 90.0, "E": 180.0, "W": 0.0}


class LocalizadorAruco:
    """Localizacion absoluta con marcadores fiduciales (reemplaza a AAY.py).

    Cada marcador DICT_4X4_50 se ancla a una CELDA + PARED con su cara
    mirando hacia ADENTRO de esa celda; las dos caras de un mismo muro fisico
    son dos registros distintos (muro entre 12 y 13: cara sur = celda 12 /
    pared N, cara norte = celda 13 / pared S). De ese anclaje se deriva la
    pose mundial del marcador, y con solvePnP (misma calibracion que la
    deteccion de obstaculos) la pose absoluta del robot en el laberinto.

    La asignacion se edita desde la web (topico comandos.marcadores), se
    valida completa o se rechaza completa, y PERSISTE en marcadores.json.
    """

    def __init__(self, camera_matrix, dist_coeffs, dist_max_cm,
                 correcciones_seg, ruta_archivo):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.dist_max_cm = float(dist_max_cm)
        # Acelerador: como maximo `correcciones_seg` sobreescrituras de pose
        # por segundo (0 = sin limite).
        self.periodo_correccion = (1.0 / correcciones_seg) if correcciones_seg > 0 else 0.0
        self.ruta_archivo = ruta_archivo

        self.activo = True           # toggle maestro (lo maneja la web)
        self.markers = {}            # id (int) -> {"celda": str, "pared": str}
        self._t_ultima_correccion = 0.0
        self._ultimo_dibujo = None   # (corners, ids, rvec, tvec) para el POV

        # Esquinas del cuadrado impreso en el marco del marcador, EN CM, para
        # que el tvec de solvePnP salga directamente en cm (X derecha,
        # Y arriba, Z saliendo del marcador hacia el observador).
        m = MARKER_SIZE_CM / 2.0
        self.obj_points = np.array(
            [[-m, m, 0], [m, m, 0], [m, -m, 0], [-m, -m, 0]], dtype=np.float32)

        # Adaptador multiversion de cv2.aruco (patron heredado de AAY.py):
        # API moderna con ArucoDetector vs API vieja de funciones sueltas.
        self.disponible = hasattr(cv2, "aruco")
        if not self.disponible:
            print("[ARUCO][AVISO] cv2.aruco no disponible (falta opencv-contrib): "
                  "localizacion por marcadores DESACTIVADA.")
            return
        aruco = cv2.aruco
        self._dic = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        try:
            self._detector = aruco.ArucoDetector(self._dic, aruco.DetectorParameters())
            self._params = None
        except AttributeError:
            self._detector = None
            self._params = (aruco.DetectorParameters_create()
                            if hasattr(aruco, "DetectorParameters_create")
                            else aruco.DetectorParameters())

        self.cargar_archivo()

    # ------------------------------------------------------------------
    # PERSISTENCIA (marcadores.json: la asignacion sobrevive reinicios)
    # ------------------------------------------------------------------
    def cargar_archivo(self):
        if not os.path.exists(self.ruta_archivo):
            print(f"[ARUCO] Sin {os.path.basename(self.ruta_archivo)}: "
                  "mapa de marcadores vacio (asignar desde la web).")
            return
        try:
            with open(self.ruta_archivo, "r", encoding="utf-8") as f:
                datos = json.load(f)
            self.markers = {int(m["id"]): {"celda": str(m["celda"]),
                                           "pared": str(m["pared"])}
                            for m in datos.get("markers", [])}
            print(f"[ARUCO] {len(self.markers)} marcadores cargados de "
                  f"{os.path.basename(self.ruta_archivo)}.")
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[ARUCO][AVISO] {self.ruta_archivo} corrupto ({e}): partiendo vacio.")
            self.markers = {}

    def guardar_archivo(self):
        try:
            with open(self.ruta_archivo, "w", encoding="utf-8") as f:
                json.dump({"markers": self.lista_markers()}, f, indent=2)
        except OSError as e:
            print(f"[ARUCO][AVISO] No se pudo guardar {self.ruta_archivo}: {e}")

    def lista_markers(self):
        return [{"id": mid, **self.markers[mid]} for mid in sorted(self.markers)]

    # ------------------------------------------------------------------
    # GEOMETRIA DEL ANCLAJE CELDA + PARED
    # ------------------------------------------------------------------
    @staticmethod
    def pose_mundial_marcador(celda, pared):
        """Posicion (cm) del centro del marcador y angulo (grados) de su
        normal saliente en el marco del laberinto."""
        col, fila = int(celda[0]), int(celda[1])
        cx, cy = col * CELL_SIZE, fila * CELL_SIZE
        dc, df = PAREDES_VECINO[pared]
        return (cx + dc * CELL_SIZE / 2.0,
                cy + df * CELL_SIZE / 2.0,
                PAREDES_NORMAL[pared])

    @staticmethod
    def cara_tiene_pared(celda, pared, conexiones):
        """Una cara solo es valida si ahi HAY PARED: o es perimetro (no
        existe celda vecina en esa direccion) o el borde esta cerrado
        (las celdas vecinas no estan conectadas mutuamente en el grafo)."""
        col, fila = int(celda[0]), int(celda[1])
        dc, df = PAREDES_VECINO[pared]
        vecino = f"{col + dc}{fila + df}"
        if vecino not in conexiones:
            return True  # perimetro del laberinto
        return not (vecino in conexiones.get(celda, [])
                    and celda in conexiones.get(vecino, []))

    # ------------------------------------------------------------------
    # VALIDACION Y PODA (ediciones de la web / cambios del laberinto)
    # ------------------------------------------------------------------
    def validar(self, propuesto, conexiones):
        """Devuelve (markers_normalizados, None) o (None, motivo). Igual
        espiritu que validar_grafo del planificador: cualquier error
        rechaza la edicion COMPLETA."""
        if not isinstance(propuesto, list):
            return None, "'markers' debe ser una lista"
        normalizado = {}
        caras = set()
        for item in propuesto:
            if not isinstance(item, dict):
                return None, "Cada marcador debe ser un objeto {id, celda, pared}"
            try:
                mid = int(item["id"])
                celda = str(item["celda"]).strip()
                pared = str(item["pared"]).strip().upper()
            except (KeyError, TypeError, ValueError):
                return None, f"Marcador invalido: {item}"
            if not (0 <= mid <= 49):
                return None, f"ID {mid} fuera del diccionario DICT_4X4_50 (0-49)"
            if mid in normalizado:
                return None, f"ID {mid} repetido"
            if not (len(celda) == 2 and celda.isdigit() and celda in conexiones):
                return None, f"Celda desconocida: '{celda}'"
            if pared not in PAREDES_VECINO:
                return None, f"Pared invalida: '{pared}' (usar N, S, E o W)"
            if (celda, pared) in caras:
                return None, f"Dos marcadores en la misma cara {celda}/{pared}"
            if not self.cara_tiene_pared(celda, pared, conexiones):
                return None, f"No hay pared en la cara {pared} de la celda {celda}"
            normalizado[mid] = {"celda": celda, "pared": pared}
            caras.add((celda, pared))
        return normalizado, None

    def podar_sin_pared(self, conexiones):
        """Elimina los marcadores cuya pared desaparecio (el editor de la web
        abrio ese pasillo). Devuelve la lista de ids eliminados."""
        eliminados = [mid for mid, cfg in self.markers.items()
                      if not self.cara_tiene_pared(cfg["celda"], cfg["pared"], conexiones)]
        for mid in eliminados:
            cfg = self.markers.pop(mid)
            print(f"[ARUCO] Marcador {mid} eliminado: ya no hay pared en "
                  f"{cfg['celda']}/{cfg['pared']}.")
        if eliminados:
            self.guardar_archivo()
        return sorted(eliminados)

    # ------------------------------------------------------------------
    # DETECCION Y POSE (solvePnP -> marco del laberinto -> centro del robot)
    # ------------------------------------------------------------------
    def detectar(self, frame):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self._dic,
                                                      parameters=self._params)
        return corners, ids

    def pose_robot_desde_pnp(self, rvec, tvec, celda, pared):
        """Pose del CENTRO DEL ROBOT en el marco del laberinto a partir del
        PnP de un marcador anclado en (celda, pared).

        Convenciones OpenCV: solvePnP entrega R, t tales que
        P_camara = R*P_marcador + t. La posicion de la LENTE en el marco del
        marcador es C = -R_t*t y el eje optico de la camara es la tercera
        fila de R. El marcador esta vertical en la pared con su normal
        saliente en el angulo `phi` del laberinto, asi que sus ejes en el
        mundo son X=(-sin phi, cos phi), Z=(cos phi, sin phi) e Y vertical
        (la componente vertical se descarta: la navegacion es 2D, y por lo
        mismo el pitch de la camara no contamina el rumbo)."""
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        C = -R.T @ t          # lente en el marco del marcador (z = al frente)
        eje_optico = R[2, :]  # eje Z de la camara, en el marco del marcador

        x_m, y_m, phi_deg = self.pose_mundial_marcador(celda, pared)
        phi = math.radians(phi_deg)
        sin_p, cos_p = math.sin(phi), math.cos(phi)

        x_lente = x_m - sin_p * C[0] + cos_p * C[2]
        y_lente = y_m + cos_p * C[0] + sin_p * C[2]
        # theta: grados antihorario desde el Este (convencion de theta_cam)
        theta = math.degrees(math.atan2(
            cos_p * eje_optico[0] + sin_p * eje_optico[2],
            -sin_p * eje_optico[0] + cos_p * eje_optico[2])) % 360.0

        # Offset camara -> robot: el centro de giro esta OFFSET_CAMARA_CM
        # DETRAS de la lente, sobre la linea del rumbo (theta no cambia:
        # es una traslacion del mismo cuerpo rigido).
        rad = math.radians(theta)
        x_robot = x_lente - OFFSET_CAMARA_CM * math.cos(rad)
        y_robot = y_lente - OFFSET_CAMARA_CM * math.sin(rad)
        return x_robot, y_robot, theta

    def _correccion_permitida(self, ahora):
        return (not self.periodo_correccion
                or (ahora - self._t_ultima_correccion) >= self.periodo_correccion)

    def procesar(self, frame, ahora):
        """Detecta marcadores en el frame y devuelve la pose absoluta del
        robot lista para publicar (dict), o None. Siempre guarda la ultima
        deteccion para anotar el video POV, aunque la correccion este
        desactivada (util para debug visual)."""
        if not self.disponible:
            return None
        corners, ids = self.detectar(frame)
        self._ultimo_dibujo = None
        if ids is None or len(ids) == 0:
            return None

        mejor = None  # (dist_cm, id, rvec, tvec, cfg): el marcador mas cercano
        for i, mid in enumerate(ids.flatten()):
            cfg = self.markers.get(int(mid))
            if cfg is None:
                continue  # marcador fisico sin asignacion en el mapa
            ok, rvec, tvec = cv2.solvePnP(self.obj_points, corners[i][0],
                                          self.K, self.dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            dist_cm = float(np.linalg.norm(tvec))
            if mejor is None or dist_cm < mejor[0]:
                mejor = (dist_cm, int(mid), rvec, tvec, cfg)

        self._ultimo_dibujo = (corners, ids,
                               mejor[2] if mejor else None,
                               mejor[3] if mejor else None)

        if mejor is None or not self.activo:
            return None
        dist_cm, mid, rvec, tvec, cfg = mejor
        if dist_cm > self.dist_max_cm:
            return None
        if not self._correccion_permitida(ahora):
            return None

        x, y, theta = self.pose_robot_desde_pnp(rvec, tvec, cfg["celda"], cfg["pared"])
        self._t_ultima_correccion = ahora
        return {"x": round(x, 1), "y": round(y, 1), "theta": round(theta, 1),
                "marker_id": mid, "dist_cm": round(dist_cm, 1), "timestamp": ahora}

    def dibujar(self, frame):
        """Anota los marcadores de la ultima deteccion sobre el frame que va
        al streaming POV."""
        if not self.disponible or not self._ultimo_dibujo:
            return
        corners, ids, rvec, tvec = self._ultimo_dibujo
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if rvec is not None:
            cv2.drawFrameAxes(frame, self.K, self.dist, rvec, tvec,
                              MARKER_SIZE_CM * 0.75)


class DetectorObstaculos:
    """Maquina de estados de obstaculos + streaming de video hacia la web."""

    ESTADO_LIBRE = "LIBRE"
    ESTADO_OBSTACULO = "OBSTACULO"

    def __init__(self, args):
        self.args = args
        dir_script = os.path.dirname(os.path.abspath(__file__))
        # UNA sola calibracion intrinseca para todo el modulo: la geometria
        # de obstaculos usa fy/cy; el PnP de ArUco usa K + dist completos.
        self.camera_matrix, self.dist_coeffs = cargar_intrinsecos(dir_script)
        self.fx, self.fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        self.cx, self.cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # Localizacion absoluta por marcadores (misma camara, misma calibracion)
        self.localizador = LocalizadorAruco(
            self.camera_matrix, self.dist_coeffs,
            dist_max_cm=args.aruco_dist_max,
            correcciones_seg=args.aruco_correcciones_seg,
            ruta_archivo=os.path.join(dir_script, ARCHIVO_MARCADORES))

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
        # === ENLACE WEB (MQTT/WebSocket) — recepción de la asignación de marcadores ArUco desde la UI ===
        client.subscribe(mqtt_topics["comandos"]["marcadores"])
        # === ENLACE WEB (MQTT/WebSocket) — recepción del toggle maestro de corrección ArUco (retenido) ===
        client.subscribe(mqtt_topics["comandos"]["marcadores_activo"])
        # Config vigente de marcadores hacia el dashboard (retenida)
        self.publicar_marcadores(ok=True, mensaje="Config de marcadores cargada")

    def _on_message(self, client, userdata, msg):
        try:
            if msg.topic == mqtt_topics["planificador"]["grafo"]:
                payload = json.loads(msg.payload.decode())
                conexiones = payload.get("graph_connections")
                if isinstance(conexiones, dict) and conexiones:
                    self.graph_connections = conexiones
                self.nodos_bloqueados = set(payload.get("nodos_bloqueados") or [])
                # Pared removida desde el editor de la web => los marcadores
                # pegados en sus caras dejan de existir (y se persiste).
                eliminados = self.localizador.podar_sin_pared(self.graph_connections)
                if eliminados:
                    self.publicar_marcadores(
                        ok=True,
                        mensaje=f"{len(eliminados)} marcador(es) eliminados por "
                                f"pared removida: {eliminados}")
                return

            if msg.topic == mqtt_topics["comandos"]["marcadores"]:
                self.aplicar_marcadores_editados(msg.payload.decode())
                return

            if msg.topic == mqtt_topics["comandos"]["marcadores_activo"]:
                activo = msg.payload.decode().strip() in ("1", "1.0", "true", "True")
                if activo != self.localizador.activo:
                    self.localizador.activo = activo
                    print(f"[ARUCO] Correccion por marcadores "
                          f"{'ACTIVADA' if activo else 'DESACTIVADA'} desde la web.")
                    self.publicar_marcadores(
                        ok=True,
                        mensaje=f"Correccion {'activada' if activo else 'desactivada'}")
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
    # MARCADORES ARUCO (edicion desde la web + eco retenido)
    # ------------------------------------------------------------------
    def aplicar_marcadores_editados(self, payload_crudo):
        """Aplica la asignacion de marcadores dibujada en la web. Patron del
        grafo del planificador: se valida TODO o se rechaza TODO, y siempre
        se responde con el eco retenido (la web confirma o revierte con el)."""
        try:
            payload = json.loads(payload_crudo)
        except json.JSONDecodeError as e:
            self.publicar_marcadores(ok=False, mensaje=f"JSON invalido: {e}")
            return
        propuesto = payload.get("markers") if isinstance(payload, dict) else None
        if propuesto is None:
            self.publicar_marcadores(ok=False, mensaje="Falta la clave 'markers'")
            return
        normalizado, motivo = self.localizador.validar(propuesto, self.graph_connections)
        if motivo:
            print(f"[ARUCO] Edicion rechazada: {motivo}")
            self.publicar_marcadores(ok=False, mensaje=motivo)
            return
        self.localizador.markers = normalizado
        self.localizador.guardar_archivo()  # persiste hasta la proxima edicion
        print(f"[ARUCO] Mapa de marcadores actualizado: {len(normalizado)} marcadores.")
        self.publicar_marcadores(ok=True,
                                 mensaje=f"{len(normalizado)} marcadores guardados")

    def publicar_marcadores(self, ok, mensaje):
        # === ENLACE WEB (MQTT/WebSocket) — eco retenido de la config de marcadores ArUco hacia el dashboard ===
        payload = {
            "markers": self.localizador.lista_markers(),
            "marker_size_cm": MARKER_SIZE_CM,
            "correccion_activa": self.localizador.activo,
            "ok": ok,
            "mensaje": mensaje,
            "timestamp": time.time(),
        }
        self._publicar(mqtt_topics["camara"]["marcadores"],
                       json.dumps(payload), retain=True)

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
        # Marcadores ArUco de la ultima deteccion (contorno + id + ejes)
        self.localizador.dibujar(frame)
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
# AUTO-VERIFICACION DE LA LOCALIZACION ARUCO (parte de --test)
# ============================================================================
def _rt_sinteticos(loc, x_robot, y_robot, theta_deg, celda, pared):
    """Modelo DIRECTO independiente para los tests: construye el (rvec, tvec)
    que solvePnP entregaria si el robot estuviera en (x_robot, y_robot,
    theta) mirando un marcador anclado en (celda, pared)."""
    th = math.radians(theta_deg)
    lente = np.array([x_robot + OFFSET_CAMARA_CM * math.cos(th),
                      y_robot + OFFSET_CAMARA_CM * math.sin(th), 0.0])
    x_m, y_m, phi_deg = loc.pose_mundial_marcador(celda, pared)
    phi = math.radians(phi_deg)
    # Ejes del marcador en el mundo (columnas: X en la pared, Y vertical,
    # Z = normal saliente hacia la celda).
    M = np.column_stack([[-math.sin(phi), math.cos(phi), 0.0],
                         [0.0, 0.0, 1.0],
                         [math.cos(phi), math.sin(phi), 0.0]])
    # Ejes de la camara en el mundo (filas; OpenCV: X derecha, Y abajo, Z optico)
    ejes_cam = np.array([[math.sin(th), -math.cos(th), 0.0],
                         [0.0, 0.0, -1.0],
                         [math.cos(th), math.sin(th), 0.0]])
    R = ejes_cam @ M                                       # marcador -> camara
    C_m = M.T @ (lente - np.array([x_m, y_m, 0.0]))        # lente en marco marcador
    t = -R @ C_m
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


def test_aruco(det):
    """Auto-verificacion sin camara ni broker: pose conocida -> rvec/tvec
    sinteticos -> reconstruccion con el codigo de produccion; mas acelerador,
    validacion de ediciones, poda por pared removida y persistencia."""
    import tempfile

    loc = det.localizador
    if not loc.disponible:
        print("  [SKIP] cv2.aruco no disponible: tests de ArUco omitidos.")
        return

    print("[TEST] Localizacion ArUco (geometria, offset -10 cm incluido):")
    casos = [
        (30.0, 32.0, 270.0, "11", "S", "de frente y centrado"),
        (24.0, 30.0, 270.0, "11", "S", "con desviacion lateral"),
        (28.0, 35.0, 250.0, "11", "S", "con theta fuera de eje"),
        (40.0, 30.0, 180.0, "11", "W", "pared Oeste, mirando al Oeste"),
        (5.0, 0.0, 180.0, "00", "W", "marcador en el perimetro"),
    ]
    for x, y, th, celda, pared, desc in casos:
        rvec, tvec = _rt_sinteticos(loc, x, y, th, celda, pared)
        xr, yr, thr = loc.pose_robot_desde_pnp(rvec, tvec, celda, pared)
        d_ang = min((thr - th) % 360.0, (th - thr) % 360.0)
        ok = abs(xr - x) < 0.01 and abs(yr - y) < 0.01 and d_ang < 0.01
        print(f"  [{'OK' if ok else 'FALLA'}] {desc}: esperado ({x:.1f},{y:.1f},{th:.1f})"
              f" -> obtenido ({xr:.2f},{yr:.2f},{thr:.2f})")

    loc._t_ultima_correccion = 100.0
    ok = (not loc._correccion_permitida(100.0 + loc.periodo_correccion * 0.5)
          and loc._correccion_permitida(100.0 + loc.periodo_correccion * 1.1))
    print(f"  [{'OK' if ok else 'FALLA'}] acelerador: respeta el periodo de "
          f"{loc.periodo_correccion:.2f} s entre correcciones")

    conex = det.graph_connections
    casos_val = [
        ([{"id": 7, "celda": "11", "pared": "S"}], True,
         "cara con pared aceptada"),
        ([{"id": 7, "celda": "12", "pared": "N"}], False,
         "cara sin pared (pasillo 12-13) rechazada"),
        ([{"id": 60, "celda": "11", "pared": "S"}], False,
         "id fuera de DICT_4X4_50 rechazado"),
        ([{"id": 7, "celda": "11", "pared": "S"},
          {"id": 7, "celda": "00", "pared": "W"}], False, "id repetido rechazado"),
        ([{"id": 7, "celda": "11", "pared": "S"},
          {"id": 8, "celda": "11", "pared": "S"}], False,
         "misma cara dos veces rechazada"),
        ([{"id": 9, "celda": "00", "pared": "W"}], True,
         "cara en el perimetro aceptada"),
    ]
    for markers, esperado, desc in casos_val:
        normalizado, motivo = loc.validar(markers, conex)
        ok = (normalizado is not None) == esperado
        print(f"  [{'OK' if ok else 'FALLA'}] validacion: {desc}")

    # Poda al abrir un pasillo + persistencia en disco (archivo temporal para
    # no tocar el marcadores.json real).
    ruta_original, markers_original = loc.ruta_archivo, loc.markers
    loc.ruta_archivo = os.path.join(tempfile.gettempdir(), "marcadores_test.json")
    loc.markers = {7: {"celda": "11", "pared": "S"},
                   8: {"celda": "10", "pared": "N"},
                   9: {"celda": "00", "pared": "W"}}
    conex_editado = {n: list(v) for n, v in conex.items()}
    conex_editado["10"].append("11")
    conex_editado["11"].append("10")
    eliminados = loc.podar_sin_pared(conex_editado)
    ok = eliminados == [7, 8] and 9 in loc.markers
    print(f"  [{'OK' if ok else 'FALLA'}] poda: abrir el pasillo 10-11 elimina los "
          f"marcadores de ambas caras (eliminados={eliminados})")

    releido = LocalizadorAruco(loc.K, loc.dist, loc.dist_max_cm, 2.0, loc.ruta_archivo)
    ok = releido.markers == loc.markers
    print(f"  [{'OK' if ok else 'FALLA'}] persistencia: marcadores.json se recarga igual")
    try:
        os.remove(loc.ruta_archivo)
    except OSError:
        pass
    loc.ruta_archivo, loc.markers = ruta_original, markers_original


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
    # Localizacion absoluta por marcadores ArUco (facilmente editables)
    parser.add_argument("--aruco-dist-max", type=float, default=50.0,
                        dest="aruco_dist_max",
                        help="cm maximos al marcador para aceptar una correccion de pose")
    parser.add_argument("--aruco-correcciones-seg", type=float, default=2.0,
                        dest="aruco_correcciones_seg",
                        help="maximo de correcciones de pose por segundo (0 = sin limite)")
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
        # Auto-verificacion de la localizacion por marcadores ArUco
        test_aruco(det)

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

            # 0) Localizacion absoluta por ArUco (misma camara y calibracion
            #    que YOLO). Si hay marcador valido dentro del alcance y el
            #    acelerador lo permite, se publica la pose para que el
            #    planificador SOBREESCRIBA su odometria.
            pose_abs = det.localizador.procesar(frame, time.time())
            if pose_abs:
                # === ENLACE WEB (MQTT/WebSocket) — pose absoluta por ArUco hacia el planificador y el dashboard ===
                det._publicar(mqtt_topics["camara"]["pose_absoluta"],
                              json.dumps(pose_abs))
                print(f"[ARUCO] Correccion: marcador {pose_abs['marker_id']} a "
                      f"{pose_abs['dist_cm']:.0f} cm -> x={pose_abs['x']:.1f} "
                      f"y={pose_abs['y']:.1f} theta={pose_abs['theta']:.1f}")

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
