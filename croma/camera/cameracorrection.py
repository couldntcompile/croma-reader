import cv2
import numpy as np

# Trapezoidal correction using 4-slider control
def apply_trapezoid_warp(image, top_skew, bottom_skew, left_skew, right_skew):
    h, w = image.shape[:2]

    max_skew_w = w * 0.4
    max_skew_h = h * 0.4

    dx_top = (top_skew - 50) / 50.0 * max_skew_w
    dx_bot = (bottom_skew - 50) / 50.0 * max_skew_w
    dy_left = (left_skew - 50) / 50.0 * max_skew_h
    dy_right = (right_skew - 50) / 50.0 * max_skew_h

    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    dst_pts = np.float32([
        [0 + dx_top, 0 + dy_left],
        [w - 1 - dx_top, 0 + dy_right],
        [w - 1 - dx_bot, h - 1 - dy_right],
        [0 + dx_bot, h - 1 - dy_left]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

# Dummy callback
def nothing(x): pass

# Set up window and sliders
cv2.namedWindow("Trapezoidal Correction")
cv2.createTrackbar("Top skew", "Trapezoidal Correction", 50, 100, nothing)
cv2.createTrackbar("Bottom skew", "Trapezoidal Correction", 50, 100, nothing)
cv2.createTrackbar("Left skew", "Trapezoidal Correction", 50, 100, nothing)
cv2.createTrackbar("Right skew", "Trapezoidal Correction", 50, 100, nothing)

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open USB camera")

print("🎛 Adjust trapezoidal correction with sliders. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Read slider values
    top = cv2.getTrackbarPos("Top skew", "Trapezoidal Correction")
    bottom = cv2.getTrackbarPos("Bottom skew", "Trapezoidal Correction")
    left = cv2.getTrackbarPos("Left skew", "Trapezoidal Correction")
    right = cv2.getTrackbarPos("Right skew", "Trapezoidal Correction")

    # Apply trapezoidal transformation
    warped = apply_trapezoid_warp(frame, top, bottom, left, right)

    # Show result
    cv2.imshow("Trapezoidal Correction", warped)

    if cv2.waitKey(10) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
