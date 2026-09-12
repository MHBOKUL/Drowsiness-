import os
import time
import urllib.request
import cv2
import numpy as np
import serial


# =========================================================
# CONFIGURATION
# =========================================================

SLEEP_THRESHOLD_SECONDS = 2.0
LOW_LIGHT_BRIGHTNESS_LIMIT = 80

# Arduino serial port
ARDUINO_PORT = "COM3"
ARDUINO_BAUDRATE = 9600

# Serial command cooldown
SERIAL_COOLDOWN = 0.5


# =========================================================
# IMAGE PROCESSING INITIALIZATION
# =========================================================

# Instantiate CLAHE once globally
clahe_obj = cv2.createCLAHE(
    clipLimit=4.0,
    tileGridSize=(8, 8)
)


# Pre-calculate gamma lookup tables
gamma_tables = {}

for g_val in [1.4, 1.8]:

    inv_gamma = 1.0 / g_val

    gamma_tables[g_val] = np.array(
        [
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]
    ).astype("uint8")


# =========================================================
# CASCADE FILE FUNCTIONS
# =========================================================

def ensure_cascade_file(filename, url):
    """
    Locate a Haar cascade file.
    If it does not exist, download it.
    """

    # OpenCV default cascade directory
    default_path = os.path.join(
        cv2.data.haarcascades,
        filename
    )

    if os.path.exists(default_path):
        return default_path

    # Local cascades directory
    fallback_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cascades"
    )

    os.makedirs(
        fallback_dir,
        exist_ok=True
    )

    local_path = os.path.join(
        fallback_dir,
        filename
    )

    if os.path.exists(local_path):
        return local_path

    try:

        print(
            f"Downloading missing cascade: {filename}"
        )

        urllib.request.urlretrieve(
            url,
            local_path
        )

        return local_path

    except Exception as e:

        raise RuntimeError(
            f"Cannot locate or download cascade "
            f"'{filename}': {e}"
        )


def load_cascade(filename, url):
    """
    Load Haar cascade classifier.
    """

    path = ensure_cascade_file(
        filename,
        url
    )

    cascade = cv2.CascadeClassifier(
        path
    )

    if cascade.empty():

        raise RuntimeError(
            f"Failed to load Haar cascade "
            f"classifier from {path}"
        )

    return cascade


# =========================================================
# CAMERA INITIALIZATION
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


# =========================================================
# LOAD FACE CASCADE
# =========================================================

face_cascade = load_cascade(
    "haarcascade_frontalface_default.xml",

    "https://raw.githubusercontent.com/opencv/"
    "opencv/master/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# LOAD EYE CASCADE
# =========================================================

eyecascade = load_cascade(
    "haarcascade_eye_tree_eyeglasses.xml",

    "https://raw.githubusercontent.com/opencv/"
    "opencv/master/data/haarcascades/"
    "haarcascade_eye_tree_eyeglasses.xml"
)


# =========================================================
# ARDUINO CONNECTION
# =========================================================

try:

    arduino = serial.Serial(
        ARDUINO_PORT,
        ARDUINO_BAUDRATE,
        timeout=1
    )

    # Allow Arduino to reset
    time.sleep(2)

    print(
        "Arduino connected successfully."
    )

    print(
        f"Port: {ARDUINO_PORT}"
    )

except Exception as e:

    print(
        f"Failed to connect to Arduino: {e}"
    )

    arduino = None


# =========================================================
# GAMMA CORRECTION
# =========================================================

def adjust_gamma(image, gamma=1.5):
    """
    Apply gamma correction using a lookup table.
    """

    table = gamma_tables.get(gamma)

    if table is None:

        inv_gamma = 1.0 / gamma

        table = np.array(
            [
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]
        ).astype("uint8")

    return cv2.LUT(
        image,
        table
    )


# =========================================================
# LOW LIGHT ENHANCEMENT
# =========================================================

def enhance_low_light(frame):
    """
    Detect low light and improve image quality.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    avg_brightness = np.mean(
        hsv[:, :, 2]
    )

    if avg_brightness < LOW_LIGHT_BRIGHTNESS_LIMIT:

        # Convert to LAB
        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(
            lab
        )

        # CLAHE
        cl = clahe_obj.apply(l)

        enhanced_lab = cv2.merge(
            (cl, a, b)
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # Dynamic gamma
        if avg_brightness < 40:

            gamma_val = 1.8

        else:

            gamma_val = 1.4

        enhanced = adjust_gamma(
            enhanced,
            gamma=gamma_val
        )

        return enhanced, True

    return frame, False


# =========================================================
# SEND COMMAND TO ARDUINO
# =========================================================

def send_to_arduino(
    sleeping,
    last_send_time,
    last_sent_state,
    force=False
):
    """
    Send sleep/awake state to Arduino.

    1 = sleeping
    0 = awake

    Newline is important because the Arduino uses:
    Serial.readStringUntil('\\n')
    """

    current_time = time.time()

    if arduino is None:
        return last_send_time, last_sent_state

    if not arduino.is_open:
        return last_send_time, last_sent_state

    # Convert boolean state to command
    if sleeping:

        command = "1"
        state_name = "SLEEPING"

    else:

        command = "0"
        state_name = "AWAKE"

    # Don't repeatedly send the same command
    if (
        not force
        and last_sent_state == sleeping
    ):
        return last_send_time, last_sent_state

    # Cooldown
    if (
        not force
        and current_time - last_send_time < SERIAL_COOLDOWN
    ):
        return last_send_time, last_sent_state

    try:

        # IMPORTANT:
        # Arduino expects newline
        arduino.write(
            (command + "\n").encode()
        )

        arduino.flush()

        print(
            f"[SERIAL OUT] Sent: "
            f"{command} ({state_name})"
        )

        return current_time, sleeping

    except Exception as e:

        print(
            f"[SERIAL ERROR] {e}"
        )

        return last_send_time, last_sent_state


# =========================================================
# MAIN VARIABLES
# =========================================================

last_send_time = 0

last_sent_state = None

eye_closed_start = None

is_sleeping = False

previous_sleeping_state = False

running = True


# =========================================================
# MAIN EXECUTION LOOP
# =========================================================

try:

    # -----------------------------------------------------
    # Start Arduino in AWAKE state
    # -----------------------------------------------------

    last_send_time, last_sent_state = send_to_arduino(
        False,
        last_send_time,
        last_sent_state,
        force=True
    )


    while running:

        # =================================================
        # READ CAMERA FRAME
        # =================================================

        ret, frame = cap.read()

        if not ret:

            print(
                "Failed to read camera frame."
            )

            break


        # =================================================
        # MIRROR IMAGE
        # =================================================

        frame = cv2.flip(
            frame,
            1
        )

        h, w, _ = frame.shape


        # =================================================
        # LOW LIGHT PROCESSING
        # =================================================

        enhanced_frame, is_low_light_active = (
            enhance_low_light(frame)
        )


        # =================================================
        # DEFAULT DISPLAY STATUS
        # =================================================

        display_status = (
            "NO FACE DETECTED"
        )

        status_color = (
            0,
            165,
            255
        )

        elapsed_closed = 0.0


        # =================================================
        # CONVERT TO GRAYSCALE
        # =================================================

        gray = cv2.cvtColor(
            enhanced_frame,
            cv2.COLOR_BGR2GRAY
        )


        # =================================================
        # FACE DETECTION
        # =================================================

        faces = face_cascade.detectMultiScale(
            gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(80, 80)
        )


        # =================================================
        # FACE FOUND
        # =================================================

        if len(faces) > 0:

            # Use largest face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, fw, fh = face


            # -------------------------------------------------
            # Draw face rectangle
            # -------------------------------------------------

            cv2.rectangle(
                frame,

                (x, y),

                (x + fw, y + fh),

                (255, 0, 0),

                2
            )


            # -------------------------------------------------
            # Face ROI
            # -------------------------------------------------

            roi_gray = gray[
                y:y + fh,
                x:x + fw
            ]

            roi_color = enhanced_frame[
                y:y + fh,
                x:x + fw
            ]


            # =================================================
            # EYE DETECTION
            # =================================================

            eyes = eyecascade.detectMultiScale(
                roi_gray,

                scaleFactor=1.1,

                minNeighbors=5,

                minSize=(20, 20)
            )


            # =================================================
            # DRAW EYE BOXES
            # =================================================

            for ex, ey, ew, eh in eyes[:2]:

                cv2.rectangle(
                    roi_color,

                    (ex, ey),

                    (ex + ew, ey + eh),

                    (0, 255, 0),

                    2
                )


            # =================================================
            # EYE STATE
            # =================================================

            current_time = time.time()


            # -------------------------------------------------
            # EYES DETECTED
            # -------------------------------------------------

            if len(eyes) >= 1:

                # Eyes open
                eye_closed_start = None

                elapsed_closed = 0.0

                is_sleeping = False

                display_status = "AWAKE"

                status_color = (
                    0,
                    255,
                    0
                )


            # -------------------------------------------------
            # NO EYES DETECTED
            # -------------------------------------------------

            else:

                # Start timer
                if eye_closed_start is None:

                    eye_closed_start = (
                        current_time
                    )


                # Calculate closed duration
                elapsed_closed = (
                    current_time
                    - eye_closed_start
                )


                # -------------------------------------------------
                # Sleep threshold reached
                # -------------------------------------------------

                if (
                    elapsed_closed
                    >= SLEEP_THRESHOLD_SECONDS
                ):

                    is_sleeping = True

                    display_status = (
                        "SLEEPING DETECTED!"
                    )

                    status_color = (
                        0,
                        0,
                        255
                    )


                # -------------------------------------------------
                # Eyes closed but not sleeping yet
                # -------------------------------------------------

                else:

                    is_sleeping = False

                    display_status = (
                        "EYES CLOSED"
                    )

                    status_color = (
                        0,
                        255,
                        255
                    )


            # =================================================
            # SEND STATE TO ARDUINO
            # =================================================

            if is_sleeping != previous_sleeping_state:

                last_send_time, last_sent_state = (
                    send_to_arduino(
                        is_sleeping,

                        last_send_time,

                        last_sent_state,

                        force=True
                    )
                )

                previous_sleeping_state = (
                    is_sleeping
                )


            # =================================================
            # EYE INFORMATION
            # =================================================

            cv2.putText(
                frame,

                f"Eyes Detected: {len(eyes)}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.putText(
                frame,

                f"Closed Time: {elapsed_closed:.1f}s",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


        # =================================================
        # NO FACE FOUND
        # =================================================

        else:

            # Reset eye timer
            eye_closed_start = None

            elapsed_closed = 0.0

            # IMPORTANT:
            # If face disappears, don't keep Arduino
            # permanently sleeping.
            is_sleeping = False

            display_status = (
                "NO FACE DETECTED"
            )

            status_color = (
                0,
                165,
                255
            )


            # -------------------------------------------------
            # Wake Arduino if necessary
            # -------------------------------------------------

            if previous_sleeping_state:

                last_send_time, last_sent_state = (
                    send_to_arduino(
                        False,

                        last_send_time,

                        last_sent_state,

                        force=True
                    )
                )

                previous_sleeping_state = False


        # =================================================
        # SYSTEM STATUS HUD
        # =================================================

        cv2.putText(
            frame,

            f"STATUS: {display_status}",

            (20, h - 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            status_color,

            2
        )


        # =================================================
        # LOW LIGHT STATUS
        # =================================================

        if is_low_light_active:

            cv2.putText(
                frame,

                "LOW LIGHT BOOST ACTIVE",

                (w - 270, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 255, 255),

                1
            )


        # =================================================
        # ARDUINO STATUS
        # =================================================

        if arduino and arduino.is_open:

            arduino_status = "ARDUINO: CONNECTED"

            arduino_color = (
                0,
                255,
                0
            )

        else:

            arduino_status = "ARDUINO: DISCONNECTED"

            arduino_color = (
                0,
                0,
                255
            )


        cv2.putText(
            frame,

            arduino_status,

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            arduino_color,

            2
        )


        # =================================================
        # DISPLAY FRAME
        # =================================================

        cv2.imshow(
            "Advanced Drowsiness Detection System",
            frame
        )


        # =================================================
        # KEYBOARD CONTROL
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        # -------------------------------------------------
        # Q = QUIT
        # -------------------------------------------------

        if key == ord("q"):

            running = False

            break


        # -------------------------------------------------
        # SPACE = FORCE AWAKE
        # -------------------------------------------------

        elif key == ord(" "):

            print(
                "[MANUAL] Force AWAKE"
            )

            is_sleeping = False

            eye_closed_start = None

            last_send_time, last_sent_state = (
                send_to_arduino(
                    False,

                    last_send_time,

                    last_sent_state,

                    force=True
                )
            )

            previous_sleeping_state = False


except KeyboardInterrupt:

    print(
        "\nInterrupted by user."
    )


except Exception as e:

    print(
        f"\nRuntime error: {e}"
    )


finally:

    # =====================================================
    # CLEANUP
    # =====================================================

    print(
        "\nShutting down..."
    )


    # -----------------------------------------------------
    # Tell Arduino to wake before closing
    # -----------------------------------------------------

    if arduino and arduino.is_open:

        try:

            arduino.write(
                b"0\n"
            )

            arduino.flush()

            time.sleep(0.2)

        except Exception:
            pass


    # -----------------------------------------------------
    # Release camera
    # -----------------------------------------------------

    cap.release()


    # -----------------------------------------------------
    # Close OpenCV windows
    # -----------------------------------------------------

    cv2.destroyAllWindows()


    # -----------------------------------------------------
    # Close Arduino
    # -----------------------------------------------------

    if arduino and arduino.is_open:

        arduino.close()

        print(
            "Arduino connection closed."
        )

    print(
        "System stopped."
    )
