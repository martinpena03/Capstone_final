# ============================================================================
# topicos.py — REGISTRO UNICO de topicos MQTT del sistema.
#
# Regla del proyecto: ningun script escribe strings de topicos a mano;
# todos importan este diccionario. La pagina web mantiene su espejo en
# 02_Pagina_web/src/lib/mqtt.js — si se agrega un topico aqui, agregarlo alla.
#
# Convenciones importantes (unica fuente de verdad):
#   * DETENCION DEL ROBOT: x_ref=-10, y_ref=-10 es la señal que la ESP32
#     interpreta como "detente". La publica laberinto_y_a_star.py mientras
#     flag_obs este activa. NO cambiar esta convencion sin tocar el firmware.
#   * flag_obs: la levanta (1) camara_y_deteccion.py al confirmar un
#     obstaculo; la limpia (0) si el obstaculo se retira o cuando el nodo
#     ya quedo bloqueado y el desvio esta en curso.
#   * nodo_bloqueado: lo publica camara_y_deteccion.py cuando un obstaculo
#     supera el timeout de re-evaluacion; laberinto_y_a_star.py lo marca
#     bloqueado en su matriz y recalcula A*.
#   * Topicos "planificador/*" son retenidos (retain=True): el broker
#     entrega el ultimo valor a la web aunque se conecte despues.
# ============================================================================

mqtt_topics = {
    "camara": {
        "x_cam": "robot/camara/x_cam",
        "y_cam": "robot/camara/y_cam",
        "theta_cam": "robot/camara/theta_cam",
        "d_pared_cam": "robot/camara/d_pared_cam",
        "nodo_actual": "robot/camara/nodo_actual",          # -> web (posicion del robot en el mapa)
        "siguiente_nodo": "robot/camara/siguiente_nodo",
        "nodo_obs": "robot/camara/nodo_obs",                # -> web (cruz parpadeante durante re-evaluacion)
        "dist_obs": "robot/camara/dist_obs",                # -> web (distancia al obstaculo, cm)
        "nodo_bloqueado": "robot/camara/nodo_bloqueado",    # camara -> laberinto (bloqueo tras timeout)
        "video": "robot/camara/video",                      # -> web (frames JPEG, POV del robot)
    },
    "planificador": {
        "grafo": "robot/planificador/grafo",                # -> web (estructura del laberinto, retenido)
        "ruta": "robot/planificador/ruta",                  # -> web (ruta A* vigente, retenido)
        "edicion": "robot/planificador/edicion",            # -> web (acuse de edicion del laberinto)
        "pose_inicial": "robot/planificador/pose_inicial",  # -> web (eco de la pose aplicada, retenido)
    },
    "telemetria": {
        "v_der": "robot/telemetria/v_der",
        "v_izq": "robot/telemetria/v_izq",
        "v_total": "robot/telemetria/v_total",
        "teta": "robot/telemetria/teta",
        "omega": "robot/telemetria/omega",
        "x": "robot/telemetria/x",
        "y": "robot/telemetria/y",
        "d_pared_der": "robot/telemetria/d_pared_der",
        "d_pared_izq": "robot/telemetria/d_pared_izq",
        "d_pared_trasera": "robot/telemetria/d_pared_trasera",
        "distancia_recorrida": "robot/telemetria/distancia_recorrida",
        "pilas": "robot/telemetria/pilas",
    },
    "estados": {
        "conexion_esp": "robot/estados/conectado_esp",
        "modo_control": "robot/estados/modo_control",
        "flag_pos": "robot/estados/flag_pos",
        "flag_obs": "robot/estados/flag_obs",               # camara -> laberinto y web (obstaculo activo)
        "flag_sen": "robot/estados/flag_sen",
        "estado_robot": "robot/estados/estado_robot",       # -> web: navegando | detenido_obstaculo | idle
        "ejecutando": "robot/estados/ejecutando",
        "grabar": "robot/estados/grabar",
        "reinicio": "robot/estados/reinicio",
    },
    "comandos": {
        "duty_der": "robot/comandos/duty_der",
        "duty_izq": "robot/comandos/duty_izq",
        "teta_ref": "robot/comandos/teta_ref",
        "v_der_ref": "robot/comandos/v_der_ref",
        "v_izq_ref": "robot/comandos/v_izq_ref",
        "v_total_ref": "robot/comandos/v_total_ref",
        "x_ref": "robot/comandos/x_ref",                    # laberinto -> esp_python_serial (nodo a nodo)
        "y_ref": "robot/comandos/y_ref",                    # laberinto -> esp_python_serial (nodo a nodo)
        "nodo_des": "robot/comandos/nodo_des",              # web -> laberinto (nodo objetivo)
        "reset_0": "robot/comandos/reset_0",
        "grafo": "robot/comandos/grafo",                    # web -> laberinto (laberinto editado)
        "pose_inicial": "robot/comandos/pose_inicial",      # web -> laberinto (nodo y orientacion de partida)
    },
}
# se usa: mqtt_topics["telemetria"]["v_der"], mqtt_topics["estados"]["conexion_esp"], mqtt_topics["comandos"]["duty_der"]
