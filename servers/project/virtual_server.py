"""Virtual server for the Traffic Signs project.

Run with:
    python launch.py --sim --task project

This file is the simulation *server* (Flask + Godot TCP). It is NOT part of
tasks/project/packages/. Those imports below are required course infrastructure —
do not remove them or simulation will not start.

    duckiebot.*     camera + wheels drivers for Godot
    launcher.*      map scene paths (project / project_signs)
    servers.common  video stream helpers
    servers.templates.project  web UI buttons

Student code (only 2 files in packages/):
    tasks.project.packages.agent          main loop + FSM + LATEST_STATUS
    tasks.project.packages.sign_detector  AprilTags (imported by agent only)

virtual_server imports agent only — never apriltag_detector / behavior_fsm etc.
"""
import argparse
import os
import socket
import sys
import threading
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

# --- Simulation plumbing (same as object_detection virtual_server) ---
from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.config import GODOT_SCENES
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.project import get_template

# --- Student project: only agent.py (+ sign_detector.py via agent) ---
import tasks.project.packages.agent as agent_module


app = Flask(__name__)

camera     = None
wheels     = None
stop_event = threading.Event()
agent_thread = None

_current_scene = 'project'
_running       = False
_manual_mode   = False
_keys_pressed  = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock     = threading.Lock()
_keys_updated  = time.time()


HTML_TEMPLATE = get_template(title='Project — Traffic Signs',
                             subtitle='Simulation')


# ---------------------------------------------------------------------------
# Frame visualization (just colour-converts; the agent already draws nothing)
# ---------------------------------------------------------------------------
def _visualize(frame_rgb):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    state = agent_module.LATEST_STATUS.get('state', '')
    if state:
        cv2.putText(bgr, f"FSM: {state}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2,
                    lineType=cv2.LINE_AA)
    return bgr


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=60)


# ---------------------------------------------------------------------------
# Manual override (so you can drive the bot around to test sign placement)
# ---------------------------------------------------------------------------
def _manual_control_loop():
    global _keys_updated
    while not stop_event.is_set():
        if not _manual_mode or wheels is None:
            time.sleep(0.05)
            continue
        if time.time() - _keys_updated > 0.5:
            with _keys_lock:
                for k in _keys_pressed:
                    _keys_pressed[k] = False
        with _keys_lock:
            kc = _keys_pressed.copy()
        left = right = 0.0
        if kc['up']:    left, right =  0.45,  0.45
        if kc['down']:  left, right = -0.45, -0.45
        if kc['up']   and kc['left']:  left, right =  0.20,  0.45
        elif kc['up'] and kc['right']: left, right =  0.45,  0.20
        elif kc['left']:               left, right = -0.30,  0.30
        elif kc['right']:              left, right =  0.30, -0.30
        try:
            if not wheels.is_game_over():
                wheels.set_wheels_speed(left, right)
        except AttributeError:
            wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
                                  hostname=socket.gethostname(),
                                  virtual=True)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    snap = dict(agent_module.LATEST_STATUS)
    snap.update({
        'running':       _running,
        'manual_mode':   _manual_mode,
        'current_scene': _current_scene,
        'game_over':     wheels.is_game_over() if wheels else False,
    })
    return jsonify(snap)


@app.route('/start', methods=['POST'])
def start():
    global _running, agent_thread, _manual_mode
    _running     = True
    _manual_mode = False
    agent_module.AGENT_RUNNING = True
    agent_module.AGENT_PAUSED = False
    if agent_thread is None or not agent_thread.is_alive():
        agent_thread = threading.Thread(
            target=agent_module.main,
            args=(camera, wheels, None, stop_event),
            daemon=True, name='ProjectAgent',
        )
        agent_thread.start()
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global _running
    _running = False
    agent_module.AGENT_RUNNING = False
    agent_module.AGENT_PAUSED = True
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    global _running
    if wheels:
        wheels.reset_game()
    _running = True
    return jsonify({'status': 'reset', 'running': _running})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global _manual_mode
    mode = (request.json or {}).get('mode', 'auto')
    _manual_mode = (mode == 'manual')
    # Tell the agent to stop driving so the manual loop owns the wheels.
    agent_module.AGENT_PAUSED = _manual_mode
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'mode': 'manual' if _manual_mode else 'auto'})


def _force_change_scene(scene_path: str, attempts: int = 5) -> bool:
    """Robustly tell Godot to switch scenes.

    Godot rebuilds its whole scene tree on a change, which closes the wheel
    TCP server. The next click on the UI hits a stale Python socket and is
    silently dropped (`[GodotWheelTransport] Send failed: WinError 10053`).
    We work around that by closing our transport, bypassing the reconnect
    throttle, reconnecting, and retrying with a short backoff so we ride
    out the scene tear-down/rebuild window.
    """
    if wheels is None:
        return False
    transport = wheels.transport
    for i in range(attempts):
        if i > 0:
            time.sleep(0.30)
        try:
            transport.close()
            transport._last_connect_attempt = 0  # bypass 1s throttle
            if not transport._ensure_connected():
                continue
            wheels.change_scene(scene_path)
            return True
        except Exception as e:
            print(f"[switch_scene] attempt {i+1} failed: {e}")
    return False


@app.route('/switch_scene', methods=['POST'])
def switch_scene():
    """Toggle between the two project maps."""
    global _current_scene
    target = (request.json or {}).get('scene', '')
    if target not in ('project', 'project_signs'):
        return jsonify({'error': f"unknown scene {target!r}"}), 400
    if target not in GODOT_SCENES:
        return jsonify({'error': f"scene {target!r} not registered"}), 400
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
        ok = _force_change_scene(GODOT_SCENES[target])
        if not ok:
            return jsonify({'error': 'Godot did not acknowledge the scene change'}), 502
    _current_scene = target
    return jsonify({'scene': target})


@app.route('/keys', methods=['POST'])
def keys():
    global _keys_updated
    data = request.json or {}
    with _keys_lock:
        for k in _keys_pressed:
            _keys_pressed[k] = bool(data.get(k, False))
    _keys_updated = time.time()
    return jsonify({'status': 'ok'})


@app.route('/remove_objects', methods=['POST'])
def remove_objects():
    """Despawn ducks / cars / signs in the current Godot scene."""
    name_filter = (request.json or {}).get('filter', '')
    if wheels and name_filter:
        try:
            wheels.remove_objects(name_filter)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'ok', 'filter': name_filter})


@app.route('/command', methods=['POST'])
def command():
    """Generic key/value command channel — kept for template compatibility."""
    data = request.json or {}
    return jsonify({'status': 'ok', 'received': data})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    global camera, wheels

    ap = argparse.ArgumentParser()
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT — TRAFFIC SIGNS  (simulation)')
    print('=' * 60)

    print('\n[1/3] Initialising wheels...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0),
        WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )

    print('\n[2/3] Initialising camera...')
    camera = GodotCameraDriver(
        godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port),
    )
    camera.start()

    print('\n[3/3] Spawning manual-control thread...')
    threading.Thread(target=_manual_control_loop, daemon=True,
                     name='ManualControl').start()

    web_port = find_available_port(args.port)
    print(f'\nWeb interface: http://localhost:{web_port}')
    print('Manual mode works without Start. Press Start for autonomous driving.')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        stop_event.set()
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
