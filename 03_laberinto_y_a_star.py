# -*- coding: utf-8 -*-
"""
laberinto_y_a_star.py — Logica de navegacion del robot de bodega.

Responsabilidad (PRD seccion B): calcular rutas optimas con A* sobre la
matriz del laberinto, enviar las referencias de posicion NODO A NODO al
puente serial (esp_python_serial.py, via MQTT), reaccionar a los avisos de
obstaculo de camara_y_deteccion.py, y mantener a la web informada del
estado, la posicion y la forma del laberinto.

Estados del robot (publicados en robot/estados/estado_robot):
    idle                a la espera de un destino desde la web; sigue
                        aceptando ediciones del laberinto.
    navegando           siguiendo la ruta A* nodo a nodo.
    detenido_obstaculo  flag_obs activa: se publican refs (-10,-10), la
                        convencion que la ESP32 interpreta como "detente".

Flujo de obstaculos (PRD A.1-A.3, la camara dirige los tiempos):
    flag_obs=1  -> detenerse (refs -10,-10) y esperar.
    flag_obs=0  -> retomar las referencias de la ruta vigente.
    nodo_bloqueado=N -> marcar N bloqueado en la matriz (persistente en la
                        sesion), recalcular A* y publicar la ruta nueva.

Ejecutar:  python 03_laberinto_y_a_star.py [--broker localhost] [--destino 54]
Smoke test sin broker:  python 03_laberinto_y_a_star.py --test
"""

import argparse
import json
import math
import threading
import time

import networkx as nx
import paho.mqtt.client as mqtt

from topicos import mqtt_topics


# ============================================================================
# GEOMETRIA DEL LABERINTO (30 nodos, celdas de 30 cm)
# Los "nodos" son los CAMINOS TRANSITABLES (celdas), no esquinas ni paredes.
# El primer digito del nombre es la columna (0-5) y el segundo la fila (0-4).
# ============================================================================
CELL_SIZE = 30  # cm entre centros de celdas contiguas

POSITIONS = {
    f"{col}{fila}": (col * CELL_SIZE, fila * CELL_SIZE)
    for col in range(6)
    for fila in range(5)
}

# Conexiones base (pasillos abiertos) del laberinto fisico. La ausencia de
# una conexion entre celdas contiguas equivale a un muro entre ellas.
GRAFO_BASE = {
    "00": ["10", "01"], "01": ["00", "02"], "02": ["01"], "03": ["04", "13"], "04": ["03", "14"],
    "10": ["00", "20"], "11": ["12", "21"], "12": ["11", "13", "22"], "13": ["03", "12"], "14": ["04", "24"],
    "20": ["10", "30"], "21": ["22", "11"], "22": ["21", "32", "12", "23"], "23": ["24", "33", "22"], "24": ["14", "23", "34"],
    "30": ["31", "20"], "31": ["30", "32"], "32": ["22", "31"], "33": ["23", "43"], "34": ["24"],
    "40": ["50", "41"], "41": ["40", "42"], "42": ["41"], "43": ["33", "44"], "44": ["43", "54"],
    "50": ["40", "51"], "51": ["50", "52"], "52": ["51", "53"], "53": ["52", "54"], "54": ["53", "44"],
}

# Orientaciones validas para la pose inicial (grados, 0 = +X Este,
# antihorario). Restringirlas a multiplos de 90 mantiene exacta la
# transformacion odometria <-> laberinto (cos/sin en {-1, 0, 1}).
ORIENTACIONES_VALIDAS = (0, 90, 180, 270)
_COS = {0: 1, 90: 0, 180: -1, 270: 0}
_SIN = {0: 0, 90: 1, 180: 0, 270: -1}

# Referencias que la ESP32 interpreta como "detener robot". Convencion ya
# implementada en el sistema de control: NO cambiar (PRD A.1).
REF_DETENCION = (-10, -10)


class NavegadorLaberinto:
    """Planificador A* + orquestacion de movimientos + enlace con la web."""

    def __init__(self, tolerancia_cm=2.0, broker="localhost", puerto=1883):
        self.tolerancia = float(tolerancia_cm)

        # Copia mutable del grafo: las ediciones de la web la reemplazan.
        self.graph_connections = {n: list(v) for n, v in GRAFO_BASE.items()}
        self.positions = dict(POSITIONS)

        # Nodos bloqueados por obstaculos confirmados. Persisten durante toda
        # la sesion (PRD C.4: la cruz queda estatica); se limpian al reiniciar.
        self.nodos_bloqueados = set()

        # Estado de navegacion
        self.estado = "idle"
        self.nodo_destino = None
        self.nodo_actual_fijo = None
        self.flag_obs_activa = False
        self._ultima_ruta_json = None
        self._ultimo_estado_publicado = None
        self._flag_pos_publicado = None

        # Telemetria cruda de la ESP (frame odometrico) via esp_python_serial.
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0  # grados, HORARIO (convencion de la ESP)

        # Pose inicial declarada desde la web. Define la transformacion
        # rigida odometria -> laberinto. Por defecto: nodo 00 mirando al
        # Este, es decir ambos marcos coinciden (comportamiento historico).
        self.origen_x = 0.0
        self.origen_y = 0.0
        self.origen_theta = 0
        self.pose_nodo = "00"

        self.broker = broker
        self.puerto = puerto
        self.client = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # TRANSFORMACION ODOMETRIA <-> LABERINTO
    # La ESP reporta (x, y) en su marco odometrico, que nace en (0,0) cada
    # vez que se resetea. Si el robot parte en otro nodo u orientado distinto
    # de 0 grados, hay que rotar y trasladar. PENDIENTE DE LAB: confirmar que
    # reset_0 pone la odometria de la ESP exactamente en (0,0,0).
    # ------------------------------------------------------------------
    def odom_a_laberinto(self, xo, yo):
        c, s = _COS[self.origen_theta], _SIN[self.origen_theta]
        return (self.origen_x + xo * c - yo * s,
                self.origen_y + xo * s + yo * c)

    def laberinto_a_odom(self, xl, yl):
        c, s = _COS[self.origen_theta], _SIN[self.origen_theta]
        dx, dy = xl - self.origen_x, yl - self.origen_y
        return (dx * c + dy * s, -dx * s + dy * c)

    def posicion_robot(self):
        """Posicion actual del robot en el marco del laberinto (cm)."""
        return self.odom_a_laberinto(self.odom_x, self.odom_y)

    # ------------------------------------------------------------------
    # GRAFO Y A*
    # ------------------------------------------------------------------
    def construir_grafo(self, excluir_bloqueados=True):
        """Grafo networkx desde graph_connections. Una arista existe solo si
        ambos nodos se declaran vecinos mutuamente (igual criterio que la
        validacion de ediciones). Los nodos bloqueados se excluyen para A*."""
        G = nx.Graph()
        for nodo in self.positions:
            G.add_node(nodo)
        for nodo, vecinos in self.graph_connections.items():
            for vecino in vecinos:
                if vecino in self.graph_connections and nodo in self.graph_connections[vecino]:
                    G.add_edge(nodo, vecino)
        if excluir_bloqueados:
            for nodo in self.nodos_bloqueados:
                if nodo in G:
                    G.remove_node(nodo)
        return G

    def heuristica(self, a, b):
        x1, y1 = self.positions[a]
        x2, y2 = self.positions[b]
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan: admisible en grilla

    def nodo_mas_cercano(self, x_cm, y_cm):
        return min(self.positions,
                   key=lambda n: math.hypot(x_cm - self.positions[n][0],
                                            y_cm - self.positions[n][1]))

    def calcular_nodo_actual(self, x_cm, y_cm):
        """Nodo exacto si el robot esta dentro de la tolerancia; si no, se
        retiene el ultimo nodo fijo (evita parpadeo entre celdas) o, como
        ultimo recurso, el mas cercano."""
        for nodo, (xt, yt) in self.positions.items():
            if abs(x_cm - xt) <= self.tolerancia and abs(y_cm - yt) <= self.tolerancia:
                return nodo
        return self.nodo_actual_fijo or self.nodo_mas_cercano(x_cm, y_cm)

    def calcular_ruta(self, origen, destino):
        """Ruta optima A* o None si el laberinto quedo sin camino viable
        (p.ej. un nodo bloqueado que parte el grafo en dos)."""
        G = self.construir_grafo()
        if origen not in G or destino not in G:
            return None
        try:
            return nx.astar_path(G, origen, destino, heuristic=self.heuristica)
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # VALIDACION Y APLICACION DE EDICIONES DESDE LA WEB (PRD C.5)
    # ------------------------------------------------------------------
    def son_adyacentes(self, a, b):
        (xa, ya), (xb, yb) = self.positions[a], self.positions[b]
        return abs(xa - xb) + abs(ya - yb) == CELL_SIZE

    def validar_grafo(self, propuesto):
        """Devuelve (grafo_normalizado, None) o (None, motivo_del_rechazo).
        Toda arista se normaliza a bidireccional; solo se aceptan aristas
        entre celdas contiguas de la grilla."""
        if not isinstance(propuesto, dict):
            return None, "El grafo debe ser un objeto"

        normalizado = {nodo: set() for nodo in self.positions}
        for nodo, vecinos in propuesto.items():
            if nodo not in self.positions:
                return None, f"Nodo desconocido: '{nodo}'"
            if not isinstance(vecinos, list):
                return None, f"Los vecinos de '{nodo}' deben ser una lista"
            for vecino in vecinos:
                if vecino not in self.positions:
                    return None, f"Vecino desconocido: '{vecino}'"
                if vecino == nodo:
                    return None, f"Arista invalida: '{nodo}' consigo mismo"
                if not self.son_adyacentes(nodo, vecino):
                    return None, f"Nodos no contiguos: '{nodo}'-'{vecino}'"
                normalizado[nodo].add(vecino)
                normalizado[vecino].add(nodo)

        return {n: sorted(v) for n, v in normalizado.items()}, None

    def aplicar_grafo_editado(self, payload_crudo):
        """Aplica un laberinto dibujado en la web. Vive solo en memoria: al
        reiniciar el proceso se vuelve al grafo base (decision consciente)."""
        try:
            payload = json.loads(payload_crudo)
        except json.JSONDecodeError as e:
            self.publicar_ack_edicion(False, f"JSON invalido: {e}")
            return

        propuesto = payload.get("graph_connections") if isinstance(payload, dict) else None
        if propuesto is None:
            self.publicar_ack_edicion(False, "Falta la clave 'graph_connections'")
            return

        normalizado, motivo = self.validar_grafo(propuesto)
        if motivo:
            print(f"[EDICION] Grafo rechazado: {motivo}")
            self.publicar_ack_edicion(False, motivo)
            self.publicar_grafo()  # la web revierte su borrador con el eco
            return

        # Swap de la referencia completa (no mutar en sitio): este callback
        # corre en el hilo de red de paho mientras el bucle principal lee.
        self.graph_connections = normalizado
        aristas = sum(len(v) for v in normalizado.values()) // 2
        print(f"[EDICION] Laberinto actualizado: {aristas} aristas.")

        self._ultima_ruta_json = None  # forzar republicacion de la ruta
        self.publicar_grafo()
        self.publicar_ack_edicion(True, f"Laberinto actualizado ({aristas} aristas)")

    # ------------------------------------------------------------------
    # POSE INICIAL DESDE LA WEB (PRD B.1)
    # ------------------------------------------------------------------
    def aplicar_pose_inicial(self, payload_crudo):
        try:
            payload = json.loads(payload_crudo)
            nodo = str(payload["nodo"]).strip()
            theta = int(payload["theta"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self.publicar_pose(ok=False, mensaje=f"Pose invalida: {e}")
            return

        if nodo not in self.positions:
            self.publicar_pose(ok=False, mensaje=f"Nodo desconocido: '{nodo}'")
            return
        if theta not in ORIENTACIONES_VALIDAS:
            self.publicar_pose(ok=False, mensaje=f"Orientacion invalida: {theta}")
            return

        with self._lock:
            self.origen_x, self.origen_y = self.positions[nodo]
            self.origen_theta = theta
            self.pose_nodo = nodo
            # La odometria local se asume en cero hasta que llegue telemetria
            # fresca posterior al reset.
            self.odom_x = self.odom_y = self.odom_theta = 0.0
            self.nodo_actual_fijo = nodo

        # Pulso de reset de odometria hacia la ESP (via esp_python_serial).
        # PENDIENTE DE LAB: confirmar la semantica exacta de reset_0 en el
        # firmware (problema #9 del diagnostico previo).
        if self.client:
            self.client.publish(mqtt_topics["comandos"]["reset_0"], "1")
            threading.Timer(
                0.5, lambda: self.client.publish(mqtt_topics["comandos"]["reset_0"], "0")
            ).start()

        self._ultima_ruta_json = None
        print(f"[POSE] Origen fijado en nodo {nodo}, theta={theta} grados.")
        self.publicar_pose(ok=True, mensaje=f"Pose aplicada: nodo {nodo}, {theta} grados")

    # ------------------------------------------------------------------
    # PUBLICACIONES HACIA LA WEB Y HACIA LA ESP
    # ------------------------------------------------------------------
    def publicar_grafo(self):
        # === ENLACE WEB (MQTT/WebSocket) — envío de la estructura del laberinto al dashboard (retenido) ===
        payload = {
            "cell_size": CELL_SIZE,
            "nodes": list(self.positions.keys()),
            "graph_connections": self.graph_connections,
            "positions": {k: list(v) for k, v in self.positions.items()},
            "nodos_bloqueados": sorted(self.nodos_bloqueados),
        }
        if self.client:
            self.client.publish(mqtt_topics["planificador"]["grafo"],
                                json.dumps(payload), retain=True)

    def publicar_ack_edicion(self, ok, mensaje):
        # === ENLACE WEB (MQTT/WebSocket) — envío del acuse de edición del laberinto al dashboard ===
        if self.client:
            self.client.publish(
                mqtt_topics["planificador"]["edicion"],
                json.dumps({"ok": ok, "mensaje": mensaje, "timestamp": time.time()}),
            )

    def publicar_pose(self, ok, mensaje):
        # === ENLACE WEB (MQTT/WebSocket) — eco de la pose inicial aplicada hacia el dashboard (retenido) ===
        payload = {
            "nodo": self.pose_nodo,
            "theta": self.origen_theta,
            "x0": self.origen_x,
            "y0": self.origen_y,
            "ok": ok,
            "mensaje": mensaje,
        }
        if self.client:
            self.client.publish(mqtt_topics["planificador"]["pose_inicial"],
                                json.dumps(payload), retain=True)

    def publicar_estado(self):
        # === ENLACE WEB (MQTT/WebSocket) — envío del estado del robot al dashboard (retenido) ===
        if self.client and self.estado != self._ultimo_estado_publicado:
            self.client.publish(mqtt_topics["estados"]["estado_robot"],
                                self.estado, retain=True)
            self._ultimo_estado_publicado = self.estado
            print(f"[ESTADO] {self.estado}")

    def publicar_flag_pos(self, valor):
        """flag_pos=1 al llegar a la meta (convencion existente: el puente
        serial lo convierte en reset_pos hacia la ESP). Solo al cambiar."""
        if self.client and valor != self._flag_pos_publicado:
            self.client.publish(mqtt_topics["estados"]["flag_pos"], str(valor))
            self._flag_pos_publicado = valor

    def publicar_refs(self, x_lab, y_lab):
        """Referencias de posicion hacia esp_python_serial (marco odometrico
        de la ESP). El envio es NODO A NODO: siempre el proximo nodo de la
        ruta, nunca la ruta completa (PRD B.3)."""
        xo, yo = self.laberinto_a_odom(x_lab, y_lab)
        if self.client:
            self.client.publish(mqtt_topics["comandos"]["x_ref"], str(int(round(xo))))
            self.client.publish(mqtt_topics["comandos"]["y_ref"], str(int(round(yo))))

    def publicar_refs_detencion(self):
        """Referencias (-10,-10): señal de detencion que la ESP ya entiende.
        No pasan por la transformacion de pose — son un codigo, no un punto."""
        if self.client:
            self.client.publish(mqtt_topics["comandos"]["x_ref"], str(REF_DETENCION[0]))
            self.client.publish(mqtt_topics["comandos"]["y_ref"], str(REF_DETENCION[1]))

    def publicar_ruta(self, nodo_siguiente, ruta_completa, x_ref, y_ref):
        # === ENLACE WEB (MQTT/WebSocket) — envío de la ruta A* vigente al dashboard (retenido) ===
        payload = {
            "nodo_actual": self.nodo_actual_fijo,
            "nodo_siguiente": nodo_siguiente,
            "nodo_destino": self.nodo_destino,
            "ruta_completa": ruta_completa,
            "nodo_bloqueado": (sorted(self.nodos_bloqueados)[-1]
                               if self.nodos_bloqueados else None),
            "nodos_bloqueados": sorted(self.nodos_bloqueados),
            "x_ref": int(round(x_ref)),
            "y_ref": int(round(y_ref)),
            "timestamp": time.time(),
        }
        # El timestamp cambia siempre: se compara sin el para deduplicar.
        clave = json.dumps({k: v for k, v in payload.items() if k != "timestamp"})
        if clave != self._ultima_ruta_json and self.client:
            self.client.publish(mqtt_topics["planificador"]["ruta"],
                                json.dumps(payload), retain=True)
            self._ultima_ruta_json = clave

    # ------------------------------------------------------------------
    # CALLBACKS MQTT
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Conectado al broker {self.broker}:{self.puerto}")
        # Telemetria de la ESP (via esp_python_serial): posicion/orientacion.
        # La orientacion se lee de aqui, NUNCA desde la web (PRD B.1).
        client.subscribe(mqtt_topics["telemetria"]["x"])
        client.subscribe(mqtt_topics["telemetria"]["y"])
        client.subscribe(mqtt_topics["telemetria"]["teta"])
        # === ENLACE WEB (MQTT/WebSocket) — recepción del nodo objetivo desde la UI ===
        client.subscribe(mqtt_topics["comandos"]["nodo_des"])
        # === ENLACE WEB (MQTT/WebSocket) — recepción del laberinto editado desde la UI ===
        client.subscribe(mqtt_topics["comandos"]["grafo"])
        # === ENLACE WEB (MQTT/WebSocket) — recepción de la pose inicial desde la UI ===
        client.subscribe(mqtt_topics["comandos"]["pose_inicial"])
        # === ENLACE WEB (MQTT/WebSocket) — recepción de la orden de desbloquear casillas (botón limpiar rastro) ===
        client.subscribe(mqtt_topics["comandos"]["limpiar_bloqueos"])
        # Señales del modulo de vision (camara_y_deteccion.py)
        client.subscribe(mqtt_topics["estados"]["flag_obs"])
        client.subscribe(mqtt_topics["camara"]["nodo_bloqueado"])

        # Publicar el grafo retenido pisa el de la sesion anterior: las
        # ediciones viven solo en memoria, la web debe ver el grafo vigente.
        self.publicar_grafo()
        self.publicar_pose(ok=True, mensaje="Pose por defecto (nodo 00, Este)")
        self._ultimo_estado_publicado = None
        self.publicar_estado()

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode().strip()

            if msg.topic == mqtt_topics["comandos"]["grafo"]:
                self.aplicar_grafo_editado(payload_str)
                return

            if msg.topic == mqtt_topics["comandos"]["pose_inicial"]:
                self.aplicar_pose_inicial(payload_str)
                return

            if msg.topic == mqtt_topics["comandos"]["limpiar_bloqueos"]:
                self.limpiar_bloqueos()
                return

            if msg.topic == mqtt_topics["comandos"]["nodo_des"]:
                if payload_str in self.positions:
                    self.nodo_destino = payload_str
                    self._ultima_ruta_json = None
                    if self.estado == "idle":
                        self.estado = "navegando"
                    print(f"[MQTT] Nuevo nodo destino: {payload_str}")
                else:
                    print(f"[MQTT] Nodo destino invalido: '{payload_str}'")
                return

            if msg.topic == mqtt_topics["estados"]["flag_obs"]:
                activa = payload_str in ("1", "1.0", "true", "True")
                if activa and not self.flag_obs_activa:
                    print("[OBSTACULO] flag_obs activa: deteniendo robot (refs -10,-10).")
                elif not activa and self.flag_obs_activa:
                    print("[OBSTACULO] flag_obs limpia: retomando la ruta vigente.")
                self.flag_obs_activa = activa
                return

            if msg.topic == mqtt_topics["camara"]["nodo_bloqueado"]:
                self.bloquear_nodo(payload_str)
                return

            # Telemetria numerica (marco odometrico de la ESP)
            valor = float(payload_str)
            if msg.topic == mqtt_topics["telemetria"]["x"]:
                self.odom_x = valor
            elif msg.topic == mqtt_topics["telemetria"]["y"]:
                self.odom_y = valor
            elif msg.topic == mqtt_topics["telemetria"]["teta"]:
                self.odom_theta = valor
        except Exception as e:
            print(f"[MQTT ERROR] {msg.topic}: {e}")

    def bloquear_nodo(self, nodo):
        """Marca un nodo como bloqueado en la matriz interna (PRD A.3).
        El bloqueo persiste toda la sesion: la web lo pinta con cruz roja
        estatica y A* deja de considerar ese nodo."""
        nodo = str(nodo).strip()
        if nodo not in self.positions:
            print(f"[BLOQUEO] Nodo desconocido: '{nodo}'")
            return
        if nodo == self.nodo_actual_fijo or nodo == self.nodo_destino:
            print(f"[BLOQUEO] Ignorado: '{nodo}' es el nodo actual o el destino.")
            return
        if nodo in self.nodos_bloqueados:
            return
        self.nodos_bloqueados.add(nodo)
        self._ultima_ruta_json = None
        print(f"[BLOQUEO] Nodo {nodo} bloqueado. Recalculando A*.")
        self.publicar_grafo()  # la web redibuja la cruz estatica

    def limpiar_bloqueos(self):
        """Desbloquea todas las casillas bloqueadas por obstaculos (lo pide
        el boton "limpiar rastro" de la web). El grafo republicado hace que
        las cruces estaticas desaparezcan y que A* vuelva a considerar esos
        nodos. Si el obstaculo sigue fisicamente ahi, la camara volvera a
        detectarlo y a bloquearlo tras el timeout, como corresponde."""
        if not self.nodos_bloqueados:
            return
        print(f"[BLOQUEO] Desbloqueando casillas: {sorted(self.nodos_bloqueados)}")
        self.nodos_bloqueados.clear()
        self._ultima_ruta_json = None
        self.publicar_grafo()

    # ------------------------------------------------------------------
    # CICLO DE NAVEGACION (bucle principal, ~10 Hz)
    # ------------------------------------------------------------------
    def paso(self):
        x_lab, y_lab = self.posicion_robot()
        self.nodo_actual_fijo = self.calcular_nodo_actual(x_lab, y_lab)

        # === ENLACE WEB (MQTT/WebSocket) — envío de la posición (nodo actual) al dashboard ===
        if self.client:
            self.client.publish(mqtt_topics["camara"]["nodo_actual"],
                                str(self.nodo_actual_fijo))
            # Pose continua en el marco del laberinto: la consume
            # camara_y_deteccion.py para ubicar los obstaculos en el mapa
            # (este proceso es el unico que conoce la pose inicial declarada).
            # theta antihorario desde el Este; la ESP reporta horario.
            theta_lab = (self.origen_theta - self.odom_theta) % 360
            self.client.publish(mqtt_topics["camara"]["x_cam"], f"{x_lab:.1f}")
            self.client.publish(mqtt_topics["camara"]["y_cam"], f"{y_lab:.1f}")
            self.client.publish(mqtt_topics["camara"]["theta_cam"], f"{theta_lab:.1f}")

        # Detencion por obstaculo: manda sobre cualquier otra cosa (PRD A.1).
        if self.flag_obs_activa and self.estado != "idle":
            self.estado = "detenido_obstaculo"
            self.publicar_refs_detencion()
            self.publicar_estado()
            return
        if self.estado == "detenido_obstaculo" and not self.flag_obs_activa:
            self.estado = "navegando" if self.nodo_destino else "idle"

        if self.estado == "idle" or not self.nodo_destino:
            # A la espera de referencias nuevas desde la web (PRD B.4). Las
            # ediciones del laberinto siguen entrando por los callbacks.
            self.publicar_estado()
            return

        # ¿Llegamos a la meta?
        x_meta, y_meta = self.positions[self.nodo_destino]
        if abs(x_lab - x_meta) <= self.tolerancia and abs(y_lab - y_meta) <= self.tolerancia:
            print(f"[A*] Meta {self.nodo_destino} alcanzada. Estado: idle.")
            self.publicar_flag_pos(1)
            self.publicar_refs(x_meta, y_meta)  # mantener posicion en la meta
            if self.client:
                self.client.publish(mqtt_topics["camara"]["siguiente_nodo"],
                                    str(self.nodo_destino))
            self.publicar_ruta(self.nodo_destino, [self.nodo_actual_fijo], x_meta, y_meta)
            self.estado = "idle"
            self.nodo_destino = None
            self.publicar_estado()
            return

        self.publicar_flag_pos(0)
        self.estado = "navegando"

        ruta = self.calcular_ruta(self.nodo_actual_fijo, self.nodo_destino)
        if ruta is None:
            # Bloqueos o ediciones dejaron el laberinto sin camino. Se retiene
            # la posicion actual y la web muestra "sin ruta viable".
            x_act, y_act = self.positions[self.nodo_actual_fijo]
            self.publicar_refs(x_act, y_act)
            self.publicar_ruta(self.nodo_actual_fijo, [], x_act, y_act)
            self.publicar_estado()
            return

        # Referencia NODO A NODO: solo el proximo nodo de la ruta. Cuando el
        # robot lo alcanza, este mismo ciclo entrega el siguiente. La
        # re-orientacion tramo a tramo la resuelve el controlador de
        # coordenadas de la ESP (giros acotados a 90 grados por diseño).
        nodo_siguiente = ruta[1] if len(ruta) > 1 else self.nodo_actual_fijo
        x_ref, y_ref = self.positions[nodo_siguiente]
        self.publicar_refs(x_ref, y_ref)
        if self.client:
            self.client.publish(mqtt_topics["camara"]["siguiente_nodo"],
                                str(nodo_siguiente))
        self.publicar_ruta(nodo_siguiente, ruta, x_ref, y_ref)
        self.publicar_estado()

    # ------------------------------------------------------------------
    def conectar(self):
        # paho-mqtt 2.x exige declarar la version del API de callbacks;
        # en la Jetson (paho 1.x) ese argumento no existe.
        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.puerto, 60)
        self.client.loop_start()

    def desconectar(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# ============================================================================
# SMOKE TEST SIN BROKER (python laberinto_y_a_star.py --test)
# ============================================================================
def modo_test():
    nav = NavegadorLaberinto()
    fallos = []

    def caso(nombre, condicion):
        print(f"  [{'OK' if condicion else 'FALLA'}] {nombre}")
        if not condicion:
            fallos.append(nombre)

    ruta = nav.calcular_ruta("00", "54")
    caso("Ruta 00->54 existe", ruta is not None and ruta[0] == "00" and ruta[-1] == "54")

    nav.nodos_bloqueados.add("22")
    ruta_bloqueada = nav.calcular_ruta("00", "54")
    caso("Bloquear 22 parte el laberinto base (sin ruta, manejado sin crash)",
         ruta_bloqueada is None)
    nav.nodos_bloqueados.clear()

    ok, motivo = nav.validar_grafo({"00": ["01"], "01": ["00"]})
    caso("Grafo valido aceptado", ok is not None and motivo is None)

    ok, motivo = nav.validar_grafo({"00": ["44"]})
    caso("Arista no contigua rechazada", ok is None and motivo is not None)

    ok, motivo = nav.validar_grafo({"99": ["00"]})
    caso("Nodo inexistente rechazado", ok is None)

    # Transformacion de pose: robot en 21 mirando al Norte (90 grados).
    nav.origen_x, nav.origen_y = nav.positions["21"]
    nav.origen_theta = 90
    x, y = nav.odom_a_laberinto(10.0, 0.0)   # avanza 10 cm de frente
    caso("Pose 21/Norte: avanzar 10 cm mueve +10 en Y del laberinto",
         abs(x - 60.0) < 1e-6 and abs(y - 40.0) < 1e-6)
    xo, yo = nav.laberinto_a_odom(x, y)
    caso("Transformacion inversa consistente",
         abs(xo - 10.0) < 1e-6 and abs(yo) < 1e-6)

    print(f"\n[TEST] {'TODO OK' if not fallos else f'{len(fallos)} fallos: {fallos}'}")
    return 0 if not fallos else 1


def main():
    parser = argparse.ArgumentParser(
        description="Planificador A* y orquestador de navegacion del laberinto")
    parser.add_argument("--broker", default="localhost", help="host del broker MQTT")
    parser.add_argument("--puerto", type=int, default=1883, help="puerto del broker MQTT")
    parser.add_argument("--tolerancia", type=float, default=2.0,
                        help="cm de tolerancia para considerar alcanzado un nodo")
    parser.add_argument("--periodo", type=float, default=0.1,
                        help="segundos entre ciclos de navegacion (~10 Hz)")
    parser.add_argument("--destino", default=None,
                        help="nodo destino inicial (opcional; si se omite, se espera a la web)")
    parser.add_argument("--test", action="store_true", help="smoke test sin broker")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(modo_test())

    nav = NavegadorLaberinto(tolerancia_cm=args.tolerancia,
                             broker=args.broker, puerto=args.puerto)
    if args.destino:
        if args.destino in nav.positions:
            nav.nodo_destino = args.destino
            nav.estado = "navegando"
        else:
            print(f"[AVISO] Destino '{args.destino}' invalido; arrancando en idle.")

    nav.conectar()
    print("[INFO] laberinto_y_a_star corriendo. Ctrl+C para salir.")
    try:
        while True:
            t0 = time.time()
            nav.paso()
            resto = args.periodo - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)
    except KeyboardInterrupt:
        print("\n[INFO] Finalizado por teclado.")
    finally:
        nav.desconectar()


if __name__ == "__main__":
    main()
