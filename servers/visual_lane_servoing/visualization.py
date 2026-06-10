import cv2
import numpy as np


def create_lane_visualization(
    image: np.ndarray,
    debug_info: dict,
    pwm_left: float,
    pwm_right: float,
) -> np.ndarray:
    display_w = 320   # 2 panels × 320 = 640 px total
    h, w = image.shape[:2]
    display_h = int(h * display_w / w)
    total_w   = display_w * 2

    # ── Build all mask images at full resolution first ───────────────────────
    rm_full = debug_info.get('red_mask', np.zeros((h, w), dtype=np.uint8))

    # Red detected: show actual camera colours where red was detected, black elsewhere
    red_detected = np.zeros((h, w, 3), dtype=np.uint8)
    # image is RGB — convert to BGR for display
    bgr_src = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    red_detected[rm_full > 0] = bgr_src[rm_full > 0]
    # Draw ROI boundary on the red-detected panel
    roi_line_y_full = int(h * 0.75)
    cv2.line(red_detected, (0, roi_line_y_full), (w, roi_line_y_full), (0, 180, 255), 2)

    # Yellow mask: yellow on black
    ym = debug_info['yellow_mask']
    yellow_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    yellow_bgr[:, :, 1] = ym
    yellow_bgr[:, :, 2] = ym

    # White mask: white on black
    wm = debug_info['white_mask']
    white_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    white_bgr[:, :, 0] = wm
    white_bgr[:, :, 1] = wm
    white_bgr[:, :, 2] = wm

    # ── Panel 1 (top-left): Camera with lane dots ────────────────────────────
    cam = cv2.resize(bgr_src, (display_w, display_h))
    scale_y = display_h / h
    scale_x = display_w / w

    for sy in debug_info.get('slice_ys', []):
        cv2.line(cam, (0, int(sy * scale_y)), (display_w, int(sy * scale_y)), (0, 255, 255), 1)

    for i, x in enumerate(debug_info.get('yellow_xs', [])):
        sy_list = debug_info.get('slice_ys', [])
        if i < len(sy_list):
            cv2.circle(cam, (int(x * scale_x), int(sy_list[i] * scale_y)), 6, (0, 255, 255), -1)

    for i, x in enumerate(debug_info.get('white_xs', [])):
        sy_list = debug_info.get('slice_ys', [])
        if i < len(sy_list):
            cv2.circle(cam, (int(x * scale_x), int(sy_list[i] * scale_y)), 6, (255, 255, 255), -1)

    # ── Panel 2 (top-right): Red detections on camera frame ──────────────────
    red_vis = cv2.resize(red_detected, (display_w, display_h))
    roi_line_y = int(display_h * 0.75)
    cv2.line(red_vis, (0, roi_line_y), (display_w, roi_line_y), (0, 180, 255), 1)
    if debug_info.get('red_line'):
        cv2.rectangle(red_vis, (0, 0), (display_w - 1, display_h - 1), (0, 0, 255), 3)

    # ── Panel 3 (bottom-left): White mask ────────────────────────────────────
    white_vis = cv2.resize(white_bgr, (display_w, display_h))

    # ── Panel 4 (bottom-right): Yellow mask ──────────────────────────────────
    yellow_vis = cv2.resize(yellow_bgr, (display_w, display_h))

    grid = np.vstack([np.hstack([cam,       red_vis]),
                      np.hstack([white_vis, yellow_vis])])

    font  = cv2.FONT_HERSHEY_SIMPLEX
    red_px        = debug_info.get('red_px', 0)
    red_label_col = (0, 0, 255) if debug_info.get('red_line') else (180, 180, 180)

    cv2.putText(grid, "Camera",                       (10,              20), font, 0.5, (0, 255, 0),     1)
    cv2.putText(grid, f"Red Detected  px:{red_px}",   (display_w + 10,  20), font, 0.5, red_label_col,   1)
    cv2.putText(grid, "White Mask",                   (10,              display_h + 20), font, 0.5, (200, 200, 200), 1)
    cv2.putText(grid, "Yellow Mask",                  (display_w + 10,  display_h + 20), font, 0.5, (0, 220, 220),  1)

    info = _info_strip(total_w, debug_info, pwm_left, pwm_right)
    return np.vstack([grid, info])


def _draw_bar(canvas, label, x0, y, bar_w, bar_h, value, font):
    cv2.putText(canvas, label, (10, y + 13), font, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
    fill = int(bar_w * np.clip(abs(value), 0, 1))
    color = (100, 100, 255) if value >= 0 else (255, 100, 100)
    cv2.rectangle(canvas, (x0, y), (x0 + fill, y + bar_h), color, -1)
    cv2.putText(canvas, f"{value:.3f}", (x0 + bar_w + 10, y + 13), font, 0.4, (200, 200, 200), 1)


def _info_strip(width, debug_info, pwm_left, pwm_right):
    h = 120
    canvas = np.zeros((h, width, 3), dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    bar_x  = 80
    bar_w  = max(100, width - bar_x - 120)
    bar_h  = 20

    # Lateral error bar
    err = debug_info['lateral_error']
    cv2.putText(canvas, "Error:", (10, 25), font, 0.5, (255, 255, 255), 1)
    cv2.rectangle(canvas, (bar_x, 5), (bar_x + bar_w, 25), (50, 50, 50), -1)
    cx = bar_x + bar_w // 2
    cv2.line(canvas, (cx, 5), (cx, 25), (100, 100, 100), 1)
    ep = int(np.clip(cx + err * bar_w / 2, bar_x, bar_x + bar_w))
    ecol = (0, 255, 0) if abs(err) < 0.1 else (0, 255, 255) if abs(err) < 0.3 else (0, 0, 255)
    cv2.circle(canvas, (ep, 15), 8, ecol, -1)
    cv2.putText(canvas, f"{err:.2f}", (bar_x + bar_w + 10, 20), font, 0.4, (200, 200, 200), 1)

    # PWM bars
    _draw_bar(canvas, "Left:",  bar_x, 40, bar_w, bar_h, pwm_left,  font)
    _draw_bar(canvas, "Right:", bar_x, 70, bar_w, bar_h, pwm_right, font)

    # Status
    if debug_info.get('red_line'):
        cv2.putText(canvas, "STOP LINE", (20, 105), font, 0.5, (0, 0, 255), 2)
    else:
        detected = debug_info['lane_detected']
        cv2.putText(canvas, "LANE OK" if detected else "NO LANE",
                    (20, 105), font, 0.5, (0, 255, 0) if detected else (0, 165, 255), 1)
    cv2.putText(canvas, f"px:{debug_info['total_lane_pixels']}  f:{debug_info.get('frame_count',0)}",
                (300, 105), font, 0.4, (200, 200, 200), 1)

    return canvas


# def create_lane_visualization(image, debug_info, pwm_left, pwm_right) -> np.ndarray:
#     from tasks.visual_lane_servoing.packages import visual_servoing_activity as student

#     display_w = 320
#     h, w = image.shape[:2]
#     display_h = int(h * display_w / w)

#     cam = cv2.resize(image, (display_w, display_h))
#     roi_y = int(display_h * 0.50)
#     cv2.line(cam, (0, roi_y), (display_w, roi_y), (0, 255, 255), 1)

#     lane_vis   = cv2.resize(cv2.applyColorMap(debug_info['lane_mask'],   cv2.COLORMAP_HOT),    (display_w, display_h))
#     white_vis  = cv2.resize(cv2.applyColorMap(debug_info['white_mask'],  cv2.COLORMAP_BONE),   (display_w, display_h))
#     ym = debug_info['yellow_mask']
#     yellow_bgr = np.zeros((*ym.shape, 3), dtype=np.uint8)
#     yellow_bgr[:, :, 1] = ym  # green channel
#     yellow_bgr[:, :, 2] = ym  # red channel  (green + red = yellow in BGR)
#     yellow_vis = cv2.resize(yellow_bgr, (display_w, display_h))

#     # Steer matrix panels — red=turn right, blue=turn left
#     mat_shape = (h, w)
#     left_mat_vis  = _draw_steer_matrix(mat_shape, student.get_steer_matrix_left_lane_markings,  display_w, display_h)
#     right_mat_vis = _draw_steer_matrix(mat_shape, student.get_steer_matrix_right_lane_markings, display_w, display_h)

#     # Add labels
#     font, green = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
#     cv2.putText(left_mat_vis,  "Left Matrix",  (10, 20), font, 0.5, (255, 255, 255), 1)
#     cv2.putText(right_mat_vis, "Right Matrix", (10, 20), font, 0.5, (255, 255, 255), 1)

#     grid = np.vstack([
#         np.hstack([cam,       lane_vis]),
#         np.hstack([white_vis, yellow_vis]),
#         np.hstack([left_mat_vis, right_mat_vis]),  # new row
#     ])

#     cv2.putText(grid, "Camera",       (10,             20), font, 0.5, green, 1)
#     cv2.putText(grid, "Lane Mask",    (display_w + 10, 20), font, 0.5, green, 1)
#     cv2.putText(grid, "White Lines",  (10,             display_h + 20), font, 0.5, green, 1)
#     cv2.putText(grid, "Yellow Lines", (display_w + 10, display_h + 20), font, 0.5, green, 1)

#     info = _info_strip(display_w * 2, debug_info, pwm_left, pwm_right)
#     return np.vstack([grid, info])


# def _draw_bar(canvas, label, x0, y, bar_w, bar_h, value, font):
#     cv2.putText(canvas, label, (10, y + 13), font, 0.5, (255, 255, 255), 1)
#     cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
#     fill = int(bar_w * np.clip(abs(value), 0, 1))
#     color = (100, 100, 255) if value >= 0 else (255, 100, 100)
#     cv2.rectangle(canvas, (x0, y), (x0 + fill, y + bar_h), color, -1)
#     cv2.putText(canvas, f"{value:.3f}", (x0 + bar_w + 10, y + 13), font, 0.4, (200, 200, 200), 1)


# def _info_strip(width, debug_info, pwm_left, pwm_right):
#     h = 120
#     canvas = np.zeros((h, width, 3), dtype=np.uint8)
#     font   = cv2.FONT_HERSHEY_SIMPLEX
#     bar_x, bar_w, bar_h = 80, 280, 20

#     # Lateral error bar
#     err = debug_info['lateral_error']
#     cv2.putText(canvas, "Error:", (10, 25), font, 0.5, (255, 255, 255), 1)
#     cv2.rectangle(canvas, (bar_x, 5), (bar_x + bar_w, 25), (50, 50, 50), -1)
#     cx = bar_x + bar_w // 2
#     cv2.line(canvas, (cx, 5), (cx, 25), (100, 100, 100), 1)
#     ep = int(np.clip(cx + err * bar_w / 2, bar_x, bar_x + bar_w))
#     ecol = (0, 255, 0) if abs(err) < 0.1 else (0, 255, 255) if abs(err) < 0.3 else (0, 0, 255)
#     cv2.circle(canvas, (ep, 15), 8, ecol, -1)
#     cv2.putText(canvas, f"{err:.2f}", (bar_x + bar_w + 10, 20), font, 0.4, (200, 200, 200), 1)

#     # PWM bars
#     _draw_bar(canvas, "Left:",  bar_x, 40, bar_w, bar_h, pwm_left,  font)
#     _draw_bar(canvas, "Right:", bar_x, 70, bar_w, bar_h, pwm_right, font)

#     # Status
#     detected = debug_info['lane_detected']
#     cv2.putText(canvas, "LANE OK" if detected else "NO LANE",
#                 (20, 105), font, 0.5, (0, 255, 0) if detected else (0, 0, 255), 1)
#     cv2.putText(canvas, f"px:{debug_info['total_lane_pixels']}  f:{debug_info.get('frame_count',0)}",
#                 (300, 105), font, 0.4, (200, 200, 200), 1)

#     return canvas

def _draw_steer_matrix(shape, matrix_fn, display_w, display_h):
    mat = matrix_fn(shape)
    # Normalize to 0-255 for display
    pos = np.clip(mat, 0, None)
    neg = np.clip(-mat, 0, None)
    vis = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    vis[:, :, 2] = (pos / pos.max() * 255).astype(np.uint8) if pos.max() > 0 else 0  # red = turn right
    vis[:, :, 0] = (neg / neg.max() * 255).astype(np.uint8) if neg.max() > 0 else 0  # blue = turn left
    return cv2.resize(vis, (display_w, display_h))