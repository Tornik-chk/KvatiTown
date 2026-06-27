import sys
import os
import signal
import threading
import time
import queue
import socket

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.sign_detection.packages.agent_with_signs import LaneServoingAgentWithSigns as LaneServoingAgent
from tasks.sign_detection.packages.sign_behavior_config import SignBehaviorConfig
from tasks.sign_detection.packages.detection import (
    should_stop as should_stop_sign_detection,
    detect_obstacles,
)

from servers.object_detection.visualization import draw_detections, draw_status_overlay
from tasks.object_detection.packages.agent import ObjectDetectionAgent, CLASS_NAMES
from servers.templates.sign_detection import SIGN_DETECTION_TEMPLATE as HTML_TEMPLATE

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

app = Flask(__name__)

lane_agent = None
det_agent = None
camera = None
wheels = None

running = False
manual_mode = False
stop_event = threading.Event()

_frame_queue = queue.Queue(maxsize=1)
_last_detections = []
_detection_lock = threading.Lock()

_stopped_by_det = False
_stop_reason = ""

keys_pressed = {"up": False, "down": False, "left": False, "right": False}
_keys_lock = threading.Lock()
_keys_last_update = time.time()

_last_pwm_left = 0.0
_last_pwm_right = 0.0
_last_sign_state = None
_last_frame_time = 0.0
_last_status_error = ""


def _set_sign_timer_paused(paused: bool):
    if lane_agent is None:
        return

    possible_objects = [
        lane_agent,
        getattr(lane_agent, "sign_behavior", None),
        getattr(lane_agent, "sign_fsm", None),
        getattr(lane_agent, "fsm", None),
        getattr(lane_agent, "_sign_behavior", None),
        getattr(lane_agent, "_sign_fsm", None),
        getattr(lane_agent, "_fsm", None),
    ]

    for obj in possible_objects:
        if obj is None:
            continue

        fn = getattr(obj, "set_timer_paused", None)
        if callable(fn):
            fn(bool(paused))
            return


def visualize(frame_bgr):
    global _stopped_by_det, _stop_reason, _last_detections
    global _last_pwm_left, _last_pwm_right, _last_sign_state
    global _last_frame_time, _last_status_error

    _last_frame_time = time.time()
    _last_status_error = ""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # IMPORTANT:
    # Always queue frames, even if the ONNX model is not loaded.
    # The color/HSV truck detector does not need the ONNX model.
    try:
        _frame_queue.put_nowait(frame_rgb)
    except queue.Full:
        pass
    except Exception as e:
        print(f"[Server] frame queue error: {e}")

    if wheels is None:
        return draw_status_overlay(frame_bgr, "Initializing...")

    with _detection_lock:
        detections = list(_last_detections)

    if manual_mode:
        _set_sign_timer_paused(True)
        _stopped_by_det = False
        _stop_reason = ""
        _last_sign_state = "MANUAL"
        return frame_bgr

    if lane_agent is not None:
        try:
            _set_sign_timer_paused(bool(_stopped_by_det))

            result = lane_agent.compute_commands(frame_rgb, detections)

            if len(result) == 3:
                pwm_left, pwm_right, state = result
            else:
                pwm_left, pwm_right = result
                state = getattr(lane_agent, "sign_state", None)

            _last_pwm_left = float(pwm_left)
            _last_pwm_right = float(pwm_right)
            _last_sign_state = state

            _draw_lane_debug(frame_bgr)

            should_stop, reason = _should_stop(detections, state)

            _stopped_by_det = bool(should_stop)
            _stop_reason = reason or ""

            print(bool(should_stop))
            _set_sign_timer_paused(bool(should_stop))

            if running and not should_stop and wheels is not None:
                wheels.set_wheels_speed(pwm_left, pwm_right)
            elif wheels is not None:
                wheels.set_wheels_speed(0.0, 0.0)

            if detections:
                draw_detections(frame_bgr, detections)

        except Exception as e:
            _last_status_error = f"lane_agent error: {e}"
            print(f"[Server] {_last_status_error}")

            _last_pwm_left = 0.0
            _last_pwm_right = 0.0
            _stopped_by_det = True
            _stop_reason = _last_status_error

            _set_sign_timer_paused(True)

            if wheels is not None:
                wheels.set_wheels_speed(0.0, 0.0)

    return frame_bgr


def _draw_lane_debug(frame_bgr):
    if lane_agent is None:
        return

    dbg = getattr(lane_agent, "last_debug_info", {}) or {}

    slice_ys = dbg.get("slice_ys", []) or []
    yellow_xs = dbg.get("yellow_xs", []) or []
    white_xs = dbg.get("white_xs", []) or []

    h, w = frame_bgr.shape[:2]

    for y in slice_ys:
        y = int(y)
        if 0 <= y < h:
            cv2.line(frame_bgr, (0, y), (w - 1, y), (90, 90, 90), 1)
            cv2.putText(
                frame_bgr,
                f"slice y={y}",
                (5, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1,
            )

    for i, x in enumerate(yellow_xs):
        if i >= len(slice_ys):
            break

        x = int(x)
        y = int(slice_ys[i])

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(
                frame_bgr,
                "Y",
                (x + 6, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )

    for i, x in enumerate(white_xs):
        if i >= len(slice_ys):
            break

        x = int(x)
        y = int(slice_ys[i])

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(
                frame_bgr,
                "W",
                (x + 6, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    count = min(len(yellow_xs), len(white_xs), len(slice_ys))

    for i in range(count):
        y = int(slice_ys[i])
        yx = int(yellow_xs[i])
        wx = int(white_xs[i])
        cx = int((yx + wx) / 2)

        if 0 <= cx < w and 0 <= y < h:
            cv2.circle(frame_bgr, (cx, y), 4, (0, 255, 0), -1)
            cv2.line(frame_bgr, (yx, y), (wx, y), (0, 180, 0), 1)


def _json_safe(value):
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return None


def _safe_list(value):
    if value is None:
        return []
    try:
        return [int(v) for v in value]
    except Exception:
        return []


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _lane_debug_payload():
    if lane_agent is None:
        return {"available": False}

    dbg = getattr(lane_agent, "last_debug_info", {}) or {}

    return {
        "available": True,
        "frame_count": _safe_int(dbg.get("frame_count", 0)),
        "lane_detected": bool(dbg.get("lane_detected", False)),
        "total_lane_pixels": _safe_int(dbg.get("total_lane_pixels", 0)),
        "lateral_error": _safe_float(dbg.get("lateral_error", 0.0)),
        "slice_ys": _safe_list(dbg.get("slice_ys", [])),
        "yellow_xs": _safe_list(dbg.get("yellow_xs", [])),
        "white_xs": _safe_list(dbg.get("white_xs", [])),
        "is_curve": bool(dbg.get("is_curve", False)),
        "curve_dir": _safe_int(dbg.get("curve_dir", 0)),
        "params": {
            "base_speed": _safe_float(getattr(lane_agent, "base_speed", 0.0)),
            "curve_speed": _safe_float(getattr(lane_agent, "curve_speed", 0.0)),
            "p_gain": _safe_float(getattr(lane_agent, "p_gain", 0.0)),
            "d_gain": _safe_float(getattr(lane_agent, "d_gain", 0.0)),
            "max_steer": _safe_float(getattr(lane_agent, "max_steer", 0.0)),
            "curve_threshold": _safe_float(getattr(lane_agent, "curve_threshold", 0.0)),
            "detection_threshold": _safe_float(getattr(lane_agent, "detection_threshold", 0.0)),
            "lane_half_width": _safe_float(getattr(lane_agent, "_lane_half_width", 0.0)),
        },
    }


def _detections_payload(dets):
    payload = []

    for bbox, score, cls_id in dets:
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1
        area = w * h
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        payload.append(
            {
                "class": CLASS_NAMES.get(cls_id, str(cls_id)),
                "class_id": int(cls_id),
                "score": round(float(score), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "center": [round(float(cx), 2), round(float(cy), 2)],
            }
        )

    return payload


def _should_stop(detections, state):
    if detections is None:
        detections = []
    return should_stop_sign_detection(detections, state)


def detection_loop():
    global _last_detections

    while not stop_event.is_set():
        try:
            frame_rgb = _frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        merged = []

        # ONNX detections.
        if det_agent is not None and getattr(det_agent, "model_loaded", False):
            try:
                result = det_agent.detect(frame_rgb)
                if result is not None:
                    merged.extend(result)
            except Exception as e:
                print(f"[DetectionLoop] ONNX detect error: {e}")

        # Color/HSV detections from sign_detection detection.py.
        # This is the part that makes the blue truck detector actually run.
        try:
            color_result = detect_obstacles(frame_rgb)
            if color_result is not None:
                merged.extend(color_result)
        except Exception as e:
            print(f"[DetectionLoop] color detect error: {e}")

        with _detection_lock:
            _last_detections = merged


def manual_control_loop():
    global _keys_last_update

    while not stop_event.is_set():
        if not manual_mode or not wheels:
            time.sleep(0.05)
            continue

        if time.time() - _keys_last_update > 0.5:
            with _keys_lock:
                for k in keys_pressed:
                    keys_pressed[k] = False

        with _keys_lock:
            kc = keys_pressed.copy()

        left = right = 0.0

        if kc["up"]:
            left, right = 0.5, 0.5

        if kc["down"]:
            left, right = -0.5, -0.5

        if kc["up"] and kc["left"]:
            left, right = 0.2, 0.5
        elif kc["up"] and kc["right"]:
            left, right = 0.5, 0.2
        elif kc["left"]:
            left, right = -0.3, 0.3
        elif kc["right"]:
            left, right = 0.3, -0.3

        wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, hostname=socket.gethostname(), virtual=False)


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with _detection_lock:
        dets = list(_last_detections)

    sign_state = None
    sign_debug = {}

    if lane_agent is not None:
        sign_state = getattr(lane_agent, "sign_state", None)
        sign_debug = getattr(lane_agent, "sign_debug", {}) or {}

    now = time.time()
    fsm = getattr(lane_agent, "_sign_fsm", None) if lane_agent else None

    return jsonify(
        {
            "hostname": socket.gethostname(),
            "time": now,
            "last_frame_age_sec": round(now - _last_frame_time, 3) if _last_frame_time else None,
            "last_error": _last_status_error,
            "running": bool(running),
            "manual_mode": bool(manual_mode),
            "camera_ready": camera is not None,
            "wheels_ready": wheels is not None,
            "agent_ready": lane_agent is not None,
            "model_loaded": det_agent.model_loaded if det_agent else False,
            "load_error": det_agent.load_error if det_agent else None,
            "trt_building": getattr(det_agent, "trt_building", False) if det_agent else False,
            "conf_threshold": det_agent.conf_threshold if det_agent else 0.5,
            "detection_count": len(dets),
            "stopped_by_detection": bool(_stopped_by_det),
            "stop_reason": _stop_reason,
            "checkpath": getattr(fsm, "_checkpath_text", "noFSM") if fsm else "Lane Agent Not loaded",
            "last_pwm": {
                "left": round(float(_last_pwm_left), 4),
                "right": round(float(_last_pwm_right), 4),
            },
            "lane_debug": _lane_debug_payload(),
            "sign_state": _json_safe(sign_state or _last_sign_state),
            "sign_debug": _json_safe(sign_debug),
            "detections": _detections_payload(dets),
            "keys_pressed": dict(keys_pressed),
        }
    )


@app.route("/start", methods=["POST"])
def start():
    global running
    running = True
    return jsonify({"status": "running"})


@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False

    _set_sign_timer_paused(True)

    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)

    return jsonify({"status": "stopped"})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    global manual_mode

    mode = request.json.get("mode", "auto") if request.json else "auto"
    manual_mode = mode == "manual"

    _set_sign_timer_paused(bool(manual_mode))

    if wheels and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)

    return jsonify({"mode": "manual" if manual_mode else "auto"})


@app.route("/keys", methods=["POST"])
def update_keys():
    global _keys_last_update

    data = request.json or {}

    with _keys_lock:
        for k in keys_pressed:
            keys_pressed[k] = bool(data.get(k, False))

    _keys_last_update = time.time()

    return jsonify({"status": "ok"})


@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    if det_agent is None:
        return jsonify({"conf_threshold": 0.5})

    data = request.json or {}
    value = data.get("conf_threshold", data.get("threshold", None))

    if value is not None:
        try:
            det_agent.conf_threshold = float(value)
        except Exception:
            pass

    return jsonify({"conf_threshold": getattr(det_agent, "conf_threshold", 0.5)})


def initialise():
    global lane_agent, det_agent, camera, wheels

    print("=" * 60)
    print("OBJECT DETECTION — LANE FOLLOW + STOP ON DETECTION")
    print("=" * 60)

    try:
        sign_cfg = SignBehaviorConfig()
        lane_agent = LaneServoingAgent(sign_config=sign_cfg)
        print(f"[Init] Lane agent ready (speed={getattr(lane_agent, 'base_speed', 'unknown')})")
        print(
            f" p_gain={getattr(lane_agent, 'p_gain', 'unknown')}, "
            f"d_gain={getattr(lane_agent, 'd_gain', 'unknown')}, "
            f"base_speed={getattr(lane_agent, 'base_speed', 'unknown')}"
        )
    except Exception as e:
        print(f"[Init] Lane agent failed: {e}")
        lane_agent = None

    try:
        det_agent = ObjectDetectionAgent()
        print(f"[Init] Detection model: {getattr(det_agent, 'model_path', None)}")
    except Exception as e:
        print(f"[Init] Detection agent failed: {e}")
        det_agent = None

    try:
        left_config = WheelPWMConfiguration()
        right_config = WheelPWMConfiguration()
        wheels = DaguWheelsDriver(left_config, right_config)
        print("[Init] Wheels ready")
    except Exception as e:
        print(f"[Init] Wheels failed: {e}")
        wheels = None

    try:
        camera = CameraDriver()
        camera.start()
        print("[Init] Camera ready")
    except Exception as e:
        print(f"[Init] Camera failed: {e}")
        camera = None

    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=manual_control_loop, daemon=True).start()


def cleanup(*_args):
    global running

    running = False
    stop_event.set()

    try:
        if wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)
    except Exception:
        pass

    try:
        shutdown_cleanup(wheels=wheels, camera=camera, stop_event=stop_event)
    except Exception:
        pass


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == "__main__":
    suppress_http_logs()
    initialise()

    port = find_available_port(5000)

    print()
    print(f"Web Interface: http://{socket.gethostname()}.local:{port}")
    print("=" * 60)
    print()

    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        cleanup()