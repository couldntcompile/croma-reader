# calibrate_camera.py
import cv2
import numpy as np
import json
from scipy.signal import find_peaks

# === Draw reference circle and segmentation rings ===
def draw_reference_circle(image, boundaries=None):
    overlay = image.copy()
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 3
    cv2.circle(overlay, center, radius, (0, 0, 255), 2)
    cv2.circle(overlay, center, 5, (0, 255, 0), -1)
    cv2.putText(overlay, "Align circle with chromatography", (center[0] - 150, center[1] + radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if boundaries:
        for r in boundaries:
            cv2.circle(overlay, center, r, (0, 255, 255), 1)
    return overlay

# === Image operations ===
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

def detect_boundaries(image, prominence):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 2 - 10
    effective_radius = int(radius * 0.95)
    polar = unwrap_chromatogram(image, center, effective_radius)
    lab = cv2.cvtColor(polar, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]  # lightness
    profile = np.mean(l_channel, axis=1)
    profile_smooth = cv2.GaussianBlur(profile[:, None], (1, 11), 0).flatten()
    peaks, _ = find_peaks(-profile_smooth, distance=10, prominence=prominence)
    return [(int((p / 360) * effective_radius)) for p in peaks]

# === UI ===
def nothing(x): pass
cv2.namedWindow("Calibration")
cv2.createTrackbar("Top skew", "Calibration", 50, 100, nothing)
cv2.createTrackbar("Bottom skew", "Calibration", 50, 100, nothing)
cv2.createTrackbar("Left skew", "Calibration", 50, 100, nothing)
cv2.createTrackbar("Right skew", "Calibration", 50, 100, nothing)
cv2.createTrackbar("Brightness", "Calibration", 100, 200, nothing)
cv2.createTrackbar("Contrast", "Calibration", 100, 300, nothing)
cv2.createTrackbar("Translate X", "Calibration", 100, 200, nothing)
cv2.createTrackbar("Translate Y", "Calibration", 100, 200, nothing)
cv2.createTrackbar("Detail", "Calibration", 5, 100, nothing)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)  # set fps explicitly if needed

if not cap.isOpened():
    raise IOError("Cannot open USB camera")

print("\n🎛 Adjust all sliders. Press SPACE to save, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Read slider values
    top = cv2.getTrackbarPos("Top skew", "Calibration")
    bottom = cv2.getTrackbarPos("Bottom skew", "Calibration")
    left = cv2.getTrackbarPos("Left skew", "Calibration")
    right = cv2.getTrackbarPos("Right skew", "Calibration")
    brightness = cv2.getTrackbarPos("Brightness", "Calibration") / 100.0
    contrast = cv2.getTrackbarPos("Contrast", "Calibration") / 100.0
    tx = cv2.getTrackbarPos("Translate X", "Calibration") - 100
    ty = cv2.getTrackbarPos("Translate Y", "Calibration") - 100
    detail = cv2.getTrackbarPos("Detail", "Calibration")

    # Apply transforms
    warped = apply_trapezoid_warp(frame, top, bottom, left, right)
    shifted = translate_image(warped, tx, ty)
    adjusted = apply_contrast(shifted, brightness, contrast)

    # Detect boundaries and draw overlay
    boundaries = detect_boundaries(adjusted, prominence=detail)
    overlay = draw_reference_circle(adjusted, boundaries)

    cv2.imshow("Calibration", overlay)
    key = cv2.waitKey(10)

    if key == 32:
        settings = {
            "top_skew": top,
            "bottom_skew": bottom,
            "left_skew": left,
            "right_skew": right,
            "brightness": brightness,
            "contrast": contrast,
            "translate_x": tx,
            "translate_y": ty,
            "detail_level": detail
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        print("\n✅ Settings saved to settings.json")
        break
    elif key == 27:
        print("\n❌ Calibration canceled.")
        break

cap.release()
cv2.destroyAllWindows()
