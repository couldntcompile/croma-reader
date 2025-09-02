# chroma_analyzer.py
import cv2
import numpy as np
import json
from scipy.signal import find_peaks

# === Load settings ===
with open("settings.json") as f:
    settings = json.load(f)

def apply_trapezoid_warp(image, top, bottom, left, right):
    h, w = image.shape[:2]
    max_w_skew = w * 0.4
    max_h_skew = h * 0.4

    dx_top = (top - 50) / 50.0 * max_w_skew
    dx_bot = (bottom - 50) / 50.0 * max_w_skew
    dy_left = (left - 50) / 50.0 * max_h_skew
    dy_right = (right - 50) / 50.0 * max_h_skew

    src_pts = np.float32([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ])
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
        "Porosity / Mineral Surface": f"{metrics[0]['grayscale_std']:.2f} texture, {metrics[0]['value']:.1f} brightness",
        "Organic-Mineral Composition": f"{metrics[1]['hue']:.1f}° hue, {metrics[1]['saturation']:.1f} sat",
        "Biological Activity": f"{metrics[2]['grayscale_std']:.2f} texture, {metrics[2]['saturation']:.1f} sat",
        "General Absorption": f"{metrics[3]['value']:.1f} brightness, {metrics[3]['hue']:.1f}° hue"
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
    overlay = draw_ring_segmentation(image_bgr, center, effective_radius, boundaries)
    return interpret_zones(metrics), overlay

# === Main loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open USB camera")

print("\n📈 Real-time chromatography analysis with adaptive zone and diameter detection. Press ESC to exit.")

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
            print(f"{k}: {v}")
        print("-" * 40)

    cv2.imshow("Chromatogram Analysis", annotated)
    if cv2.waitKey(100) == 27:
        break

cap.release()
cv2.destroyAllWindows()
