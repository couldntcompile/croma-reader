# chroma_analyzer.py
import cv2
import numpy as np
import json
import socket
import struct
from scipy.signal import find_peaks

OSC_IP = "::1"  # IPv6 localhost
OSC_PORT = 9000

NORMALIZATION_RANGES = {
    "porosity/std": (0.0, 80.0),
    "porosity/value": (50.0, 200.0),
    "organic/hue": (0.0, 180.0),
    "organic/saturation": (0.0, 255.0),
    "bio/std": (0.0, 80.0),
    "bio/saturation": (0.0, 255.0),
    "absorption/value": (0.0, 255.0),
    "absorption/hue": (0.0, 180.0),
}

with open("settings.json") as f:
    settings = json.load(f)

def pad_string(s):
    s = s.encode("utf-8") + b"\0"
    while len(s) % 4 != 0:
        s += b"\0"
    return s

def send_osc_ipv6(address, values, ip=OSC_IP, port=OSC_PORT):
    msg = pad_string(address)
    type_tag = "," + "".join("f" for _ in values)
    msg += pad_string(type_tag)
    for v in values:
        msg += struct.pack(">f", float(v))
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    sock.sendto(msg, (ip, port))
    sock.close()

def normalize(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def apply_trapezoid_warp(image, top, bottom, left, right):
    h, w = image.shape[:2]
    max_w_skew = w * 0.4
    max_h_skew = h * 0.4
    dx_top = (top - 50) / 50.0 * max_w_skew
    dx_bot = (bottom - 50) / 50.0 * max_w_skew
    dy_left = (left - 50) / 50.0 * max_h_skew
    dy_right = (right - 50) / 50.0 * max_h_skew
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = np.float32([
        [0 + dx_top, 0 + dy_left],
        [w - 1 - dx_top, 0 + dy_right],
        [w - 1 - dx_bot, h - 1 - dy_right],
        [0 + dx_bot, h - 1 - dy_left]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (w, h))

def apply_contrast(image, brightness=1.0, contrast=1.0):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1.0) * 128)

def translate_image(image, tx=0, ty=0):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h))

def unwrap_chromatogram(image, center, radius):
    return cv2.warpPolar(image, (360, radius), center, radius, cv2.WARP_POLAR_LINEAR)

def detect_radial_transitions(polar_img, num_zones=4):
    gray = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY)
    profile = np.mean(gray, axis=1)
    profile_smooth = cv2.GaussianBlur(profile[:, None], (1, 11), 0).flatten()
    peaks, _ = find_peaks(-profile_smooth, distance=10, prominence=5)
    if len(peaks) >= num_zones - 1:
        bounds = [0] + sorted(peaks[:num_zones - 1].tolist()) + [polar_img.shape[0]]
    else:
        h = polar_img.shape[0]
        step = h // num_zones
        bounds = [i * step for i in range(num_zones)] + [h]
    return [polar_img[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)], bounds[1:-1]

def extract_zone_metrics(zone):
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    return {
        "hue": np.mean(hsv[:, :, 0]),
        "saturation": np.mean(hsv[:, :, 1]),
        "value": np.mean(hsv[:, :, 2]),
        "grayscale_mean": np.mean(gray),
        "grayscale_std": np.std(gray)
    }

def interpret_zones(metrics):
    return {
        "porosity/std": metrics[0]['grayscale_std'],
        "porosity/value": metrics[0]['value'],
        "organic/hue": metrics[1]['hue'],
        "organic/saturation": metrics[1]['saturation'],
        "bio/std": metrics[2]['grayscale_std'],
        "bio/saturation": metrics[2]['saturation'],
        "absorption/value": metrics[3]['value'],
        "absorption/hue": metrics[3]['hue']
    }

def draw_ring_segmentation(image, center, effective_radius, boundaries):
    output = image.copy()
    for r in boundaries:
        scaled_r = int((r / 360) * effective_radius)
        cv2.circle(output, center, scaled_r, (0, 255, 255), 2)
    cv2.circle(output, center, effective_radius, (0, 100, 255), 2)
    return output

def detect_chromatogram_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (image.shape[1] // 2, image.shape[0] // 2), min(image.shape[:2]) // 2 - 10
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return (int(x), int(y)), int(radius)

def process_chromatogram(image_bgr):
    center, radius = detect_chromatogram_circle(image_bgr)
    effective_radius = int(radius * 0.95)
    polar = unwrap_chromatogram(image_bgr, center, effective_radius)
    zones, boundaries = detect_radial_transitions(polar, num_zones=4)
    metrics = [extract_zone_metrics(zone) for zone in zones]
    interpreted = interpret_zones(metrics)
    overlay = draw_ring_segmentation(image_bgr, center, effective_radius, boundaries)
    return interpreted, overlay

# === Main loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open USB camera")

print("\n📈 Real-time chromatography analysis with normalized OSC output. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    warped = apply_trapezoid_warp(frame, settings['top_skew'], settings['bottom_skew'], settings['left_skew'], settings['right_skew'])
    shifted = translate_image(warped, settings['translate_x'], settings['translate_y'])
    corrected = apply_contrast(shifted, settings['brightness'], settings['contrast'])

    result, annotated = process_chromatogram(corrected)
    if result:
        print("\033[2J\033[H", end="")
        print("🧪 Real-Time Soil Chromatography Interpretation:\n")
        for k, v in result.items():
            min_val, max_val = NORMALIZATION_RANGES.get(k, (0.0, 1.0))
            norm_val = normalize(v, min_val, max_val)
            print(f"/{k}: {v:.2f} (normalized: {norm_val:.2f})")
            send_osc_ipv6(f"/{k}", [norm_val])
        print("-" * 40)

#     cv2.imshow("Chromatogram Analysis", annotated)
#     if cv2.waitKey(100) == 27:
#         break

cap.release()
cv2.destroyAllWindows()
