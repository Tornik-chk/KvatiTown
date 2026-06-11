import sys
import os
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import socket
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from servers.visual_lane_servoing.visualization import create_lane_visualization
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE

from duckiebot.wheel_driver.wheels_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.camera_driver.camera_driver import CameraDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

LANE_CONFIG_FILE    = os.path.join(project_root, 'config', 'lane_servoing_config.yaml')
LANE_HSV_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_hsv_config.yaml')


def _get_student_module():
    from tasks.visual_lane_servoing.packages import visual_servoing_activity
    return visual_servoing_activity


app = Flask(__name__)

camera     = None
wheels     = None
agent      = None
running    = False
stop_event = threading.Event()

_control_thread: threading.Thread = None


def _control_loop():
    """Fixed-rate control loop: one wheel command per camera frame."""
    while not stop_event.is_set():
        if camera is None or agent is None or wheels is None:
            stop_event.wait(0.05)
            continue

        ok, frame = camera.read_rgb()
        if not ok or frame is None:
            continue

        left, right = agent.compute_commands(frame)
        if running:
            wheels.set_wheels_speed(left, right)
        else:
            wheels.set_wheels_speed(0.0, 0.0)


def visualize(frame):
    """Display-only: uses results already computed by _control_loop."""
    if agent is None or wheels is None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    debug_info = agent.last_debug_info
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return create_lane_visualization(bgr, debug_info, wheels.left_pwm, wheels.right_pwm)


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, config=agent, hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset', methods=['POST'])
def reset():
    if agent is not None:
        agent.reset()
    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'ok'})


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    agent.p_gain     = float(data.get('k_d',   agent.p_gain))
    agent.d_gain     = float(data.get('k_phi', agent.d_gain))
    agent.base_speed = float(data.get('const', agent.base_speed))
    try:
        with open(LANE_CONFIG_FILE, 'r') as f:
            saved = yaml.safe_load(f) or {}
        saved['p_gain']     = agent.p_gain
        saved['d_gain']     = agent.d_gain
        saved['base_speed'] = agent.base_speed
        with open(LANE_CONFIG_FILE, 'w') as f:
            yaml.dump(saved, f, default_flow_style=False)
    except Exception as e:
        print(f"[LaneServoing] Could not save config: {e}")
    return jsonify({'status': 'ok'})


@app.route('/get_hsv')
def get_hsv():
    return jsonify(_get_student_module().get_hsv_bounds())


@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    data = request.json
    mod = _get_student_module()
    current = mod.get_hsv_bounds()
    current.update({k: int(v) for k, v in data.items()})
    mod.set_hsv_bounds(
        [current['yellow_lower_h'], current['yellow_lower_s'], current['yellow_lower_v']],
        [current['yellow_upper_h'], current['yellow_upper_s'], current['yellow_upper_v']],
        [current['white_lower_h'],  current['white_lower_s'],  current['white_lower_v']],
        [current['white_upper_h'],  current['white_upper_s'],  current['white_upper_v']],
    )
    try:
        with open(LANE_HSV_CONFIG_FILE, 'w') as f:
            yaml.dump(current, f, default_flow_style=False)
    except Exception as e:
        print(f"[LaneServoing] Could not save HSV config: {e}")
    return jsonify({'status': 'ok'})


@app.route('/start', methods=['POST'])
def start():
    global running
    running = True
    print("[Control] Started")
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    print("[Control] Stopped")
    return jsonify({'status': 'stopped'})


@app.route('/running')
def get_running():
    return jsonify({'running': running})


@app.route('/status')
def status():
    if agent is None:
        return jsonify({'status': 'not_initialized'})
    return jsonify({
        'status': 'active',
        'frame_count': agent.frame_count,
        'config': {'p_gain': agent.p_gain, 'd_gain': agent.d_gain,
                   'base_speed': agent.base_speed,
                   'detection_threshold': agent.detection_threshold},
    })


def main():
    global camera, wheels, agent, _control_thread

    import argparse
    ap = argparse.ArgumentParser(description="Real Lane Servoing Server")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print("=" * 60)
    print("REAL LANE SERVOING SERVER")
    print("=" * 60)

    print("\n[1/3] Initializing wheels driver...")
    wheels = DaguWheelsDriver(
        WheelPWMConfiguration(pwm_min=60), WheelPWMConfiguration(pwm_min=60),
    )
    print("  Wheels: OK")

    print("\n[2/3] Initializing camera driver...")
    camera = CameraDriver()
    camera.start()
    print("  Camera: OK")

    print("\n[3/3] Creating agent...")
    agent = LaneServoingAgent()
    print(f"  p_gain={agent.p_gain}, d_gain={agent.d_gain}, base_speed={agent.base_speed}")

    _control_thread = threading.Thread(target=_control_loop, daemon=True, name="lane-control")
    _control_thread.start()
    print("  Control loop started")

    web_port = find_available_port(args.port)
    if web_port != args.port:
        print(f"  Port {args.port} busy, using {web_port}")

    print("\n" + "=" * 60)
    print(f"Web Interface: http://localhost:{web_port}")
    print("=" * 60 + "\n")

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == "__main__":
    sys.exit(main())
