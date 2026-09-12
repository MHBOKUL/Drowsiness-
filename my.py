import os
import time
import urllib.request
import cv2
import numpy as np
import serial

SLEEP_THRESHOLD_SECONDS = 2.0  # Seconds of eye closure before sleep is detected
LOW_LIGHT_BRIGHTNESS_LIMIT = 80  # Frame brightness limit for low light boost

clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

gamma_tables = {}
for g_val in [1.4, 1.8]:
    inv_gamma = 1.0 / g_val
    gamma_tables[g_val] = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")


def ensure_cascade_file(filename, url):
    """Locate a cascade XML file or download it if missing."""
    default_path = os.path.join(cv2.data.haarcascades, filename)
    if os.path.exists(default_path):
        return default_path

    fallback_dir = os.path.join(os.path.dirname(__file__), "cascades")
    os.makedirs(fallback_dir, exist_ok=True)
    local_path = os.path.join(fallback_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        print(f"Downloading missing cascade: {filename}")
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Cannot locate or download cascade file '{filename}': {e}"
        )


def load_cascade(filename, url):
    """Load a Haar cascade classifier from a reliable source."""
    path = ensure_cascade_file(filename, url)
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade classifier from {path}")
    return cascade



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = load_cascade(
    "haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
)
eyecascade = load_cascade(
    "haarcascade_eye_tree_eyeglasses.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
)

try:
    arduino = serial.Serial("COM3", 9600, timeout=1)
    time.sleep(2)
    print("Arduino connected successfully")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    arduino = None


def adjust_gamma(image, gamma=1.5):
    """Apply dynamic gamma correction using precomputed lookup tables."""
    table = gamma_tables.get(gamma)
    if table is None:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
    return cv2.LUT(image, table)


def enhance_low_light(frame):
    """Enhance contrast in low-light environments using CLAHE + Gamma."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])

    if avg_brightness < LOW_LIGHT_BRIGHTNESS_LIMIT:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe_obj.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        gamma_val = 1.8 if avg_brightness < 40 else 1.4
        enhanced = adjust_gamma(enhanced, gamma=gamma_val)
        return enhanced, True

    return frame, False


def send_to_arduino(status, last_send_time, cooldown=1.0):
    """Non-blocking signal transfer to Arduino with timestamp throttling."""
    current_time = time.time()
    if arduino and arduino.is_open:
        if current_time - last_send_time >= cooldown:
            if status:
                arduino.write(b"1")  # '1' for Sleeping
                print("[SERIAL OUT] Sent: 1 (SLEEPING)")
                time.sleep(5)  # Short delay to ensure Arduino processes the signal
            else:
                arduino.write(b"0")  # '0' for Awake
                print("[SERIAL OUT] Sent: 0 (AWAKE)")
            return current_time
    return last_send_time


last_send_time = time.time()
eye_closed_start = None
is_sleeping = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Low-Light Detection & Enhancement
        enhanced_frame, is_low_light_active = enhance_low_light(frame)

        display_status = "NO FACE DETECTED"
        status_color = (0, 165, 255)  # Orange
        elapsed_closed = 0.0  # Initialize variable to avoid UnboundLocalError

        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) > 0:
            display_status = "FACE DETECTED"
            status_color = (0, 255, 255)

            x, y, fw, fh = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
            roi_gray = gray[y : y + fh, x : x + fw]
            roi_color = enhanced_frame[y : y + fh, x : x + fw]

            eyes = eyecascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )

            for ex, ey, ew, eh in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            current_time = time.time()
            if len(eyes) >= 1:
                eye_closed_start = None
                is_sleeping = False
                elapsed_closed = 0.0
                display_status = "AWAKE"
                status_color = (0, 255, 0)
            else:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                elapsed_closed = current_time - eye_closed_start
                
                if elapsed_closed >= SLEEP_THRESHOLD_SECONDS:
                    is_sleeping = True
                    display_status = "SLEEPING DETECTED!"
                    status_color = (0, 0, 255)
                else:
                    is_sleeping = False
                    display_status = "EYES CLOSED"
                    status_color = (0, 255, 255)

            cv2.putText(
                frame,
                f"Eyes Detected: {len(eyes)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Closed Time: {elapsed_closed:.1f}s",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            last_send_time = send_to_arduino(is_sleeping, last_send_time)
        else:
            eye_closed_start = None
            is_sleeping = False

        cv2.putText(
            frame,
            f"STATUS: {display_status}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        if is_low_light_active:
            cv2.putText(
                frame,
                "LOW LIGHT BOOST ACTIVE",
                (w - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.imshow("Advanced Drowsiness Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino connection closed.")