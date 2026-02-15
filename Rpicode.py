import time
import serial
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# ---- NEW: tiny web API for Node-RED/dashboard ----
from flask import Flask, jsonify
import threading

# =======================
# Serial (Arduino) config
# =======================
SERIAL_PORT = "/dev/ttyUSB0"   
BAUD_RATE = 115200

# =======================
# DASHBOARD / COUNTERS
# =======================
RECYCLE_CAPACITY_ITEMS = 5
NONRECYCLE_CAPACITY_ITEMS = 5

recyCount = 0
nonRecyCount = 0

# This dict is what Node-RED/website will read
latest = {
    "rCount": 0,
    "nrCount": 0,
    "rPercent": 0.0,
    "nrPercent": 0.0,
    "rFull": 0,
    "nrFull": 0,
    "lastEvent": "boot",
    "lastCmd": "",
    "lastAck": "",
    "lastUpdate": 0.0
}

def clamp_percent(p):
    if p < 0: return 0.0
    if p > 100: return 100.0
    return float(p)

def update_latest(last_event=""):
    global latest, recyCount, nonRecyCount
    rPercent = clamp_percent((recyCount / RECYCLE_CAPACITY_ITEMS) * 100.0) if RECYCLE_CAPACITY_ITEMS > 0 else 0.0
    nrPercent = clamp_percent((nonRecyCount / NONRECYCLE_CAPACITY_ITEMS) * 100.0) if NONRECYCLE_CAPACITY_ITEMS > 0 else 0.0

    latest["rCount"] = int(recyCount)
    latest["nrCount"] = int(nonRecyCount)
    latest["rPercent"] = round(rPercent, 1)
    latest["nrPercent"] = round(nrPercent, 1)
    latest["rFull"] = 1 if recyCount >= RECYCLE_CAPACITY_ITEMS else 0
    latest["nrFull"] = 1 if nonRecyCount >= NONRECYCLE_CAPACITY_ITEMS else 0
    if last_event:
        latest["lastEvent"] = last_event
    latest["lastUpdate"] = time.time()

# =======================
# Web API (for Node-RED)
# =======================
app = Flask(__name__)

@app.route("/status")
def status():
    return jsonify(latest)

def start_api():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# =======================
# YOLO config (Pi-friendly)
# =======================
MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 320
CONF_THRES = 0.25
IOU_THRES = 0.45
MAX_DET = 8

# Relevant COCO classes for your project (+ "book" as paper/tissue proxy)
ALLOWED_CLASS_IDS = [39, 40, 41, 42, 43, 44, 45, 73, 79]

# =======================
# BIN LOGIC (final)
# =======================
CLASS_TO_BIN = {
    # Recyclable
    "bottle": "L",
    "cup": "L",
    "bowl": "L",
    "book": "L",        # paper/tissue proxy

    # Non-recyclable
    "fork": "R",
    "knife": "R",
    "spoon": "R",
    "wine glass": "R",
    "toothbrush": "R",
}
UNKNOWN_DEFAULT_BIN = "R"

PAPER_LIKE = {"book"}
PAPER_STREAK_NEEDED = 2

# =======================
# Camera / window config
# =======================
FRAME_W, FRAME_H = 1280, 960

# =======================
# ROI (Area of Interest)
# =======================
ROI_X1, ROI_Y1 = 0.30, 0.20
ROI_X2, ROI_Y2 = 0.70, 0.74
ROI_OVERLAP_MIN = 0.01

# =======================
# Trigger / debounce config
# =======================
DECISION_COOLDOWN_SEC = 0.15
LATCH_HOLD_SEC = 0.25
MISSING_FRAMES_TO_RESET = 8

def send_with_ack(ser, cmd: bytes, timeout=0.8, retries=4) -> (bool, bytes):
    """
    Sends b"L" or b"R" and waits for Arduino to reply EXACTLY: b"ACK:" + cmd
    Returns (ok, last_line_seen)
    """
    expected = b"ACK:" + cmd
    last_line = b""

    for _ in range(retries):
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write(cmd)
        ser.flush()

        start = time.time()
        while time.time() - start < timeout:
            if ser.in_waiting:
                line = ser.readline().strip()
                last_line = line

                if line == expected:
                    return True, line

                # If Arduino blocks because FULL, we capture it and stop retrying
                if line.startswith(b"BLOCKED:"):
                    return False, line

                if line == b"BUSY":
                    time.sleep(0.2)
                    break

    return False, last_line

def rect_intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))

def rect_area(r):
    return float(max(0, r[2] - r[0]) * max(0, r[3] - r[1]))

def bbox_overlaps_roi(bbox, roi):
    inter = rect_intersection_area(bbox, roi)
    a = rect_area(bbox)
    if a <= 0:
        return False
    return (inter / a) >= ROI_OVERLAP_MIN

def best_detection_in_roi(result, names, roi):
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0

    best_conf = 0.0
    best_label = None

    for b in result.boxes:
        conf = float(b.conf[0])
        if conf < CONF_THRES:
            continue

        cls_id = int(b.cls[0])
        label = names.get(cls_id, str(cls_id)).strip().lower()

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bbox = (x1, y1, x2, y2)

        if not bbox_overlaps_roi(bbox, roi):
            continue

        if conf > best_conf:
            best_conf = conf
            best_label = label

    return best_label, best_conf

def main():
    global recyCount, nonRecyCount

    ser = None
    picam2 = None

    latched = False
    latch_time = 0.0
    last_sent_time = 0.0
    missing = 0

    last_label = None
    streak = 0

    t0 = time.time()
    frames = 0
    no_valid_frames = 0

    # --- start web API thread immediately ----
    threading.Thread(target=start_api, daemon=True).start()
    update_latest("api_started")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)

        # ----  reset Arduino counters EVERY time script runs ----
        ser.write(b"Z")
        ser.flush()
        recyCount = 0
        nonRecyCount = 0
        latest["lastCmd"] = "Z"
        latest["lastAck"] = "COUNTERS_RESET (requested)"
        update_latest("reset_sent")

        model = YOLO(MODEL_PATH)
        names = model.names

        print("Allowed classes:")
        for cid in ALLOWED_CLASS_IDS:
            print(f"  {cid}: {names[cid]}")
        print("\nBin mapping:")
        for k, v in CLASS_TO_BIN.items():
            print(f"  {k:11s} -> {'LEFT (recycle)' if v == 'L' else 'RIGHT (trash)'}")

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.4)

        while True:
            frame = picam2.capture_array()
            frames += 1
            now = time.time()

            h, w = frame.shape[:2]
            rx1, ry1 = int(ROI_X1 * w), int(ROI_Y1 * h)
            rx2, ry2 = int(ROI_X2 * w), int(ROI_Y2 * h)
            roi = (rx1, ry1, rx2, ry2)

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.imshow("Sorting (q=quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            if now - t0 >= 2.0:
                fps = frames / (now - t0)
                print(f"[DEBUG] FPS ~ {fps:.1f} | no_valid_frames={no_valid_frames} | latched={latched}")
                t0 = now
                frames = 0

            if latched and (now - latch_time) >= LATCH_HOLD_SEC:
                latched = False

            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                classes=ALLOWED_CLASS_IDS,
                max_det=MAX_DET,
                verbose=False
            )

            label, conf = best_detection_in_roi(results[0], names, roi)

            if label is None:
                no_valid_frames += 1
                missing += 1
                last_label = None
                streak = 0
                if missing >= MISSING_FRAMES_TO_RESET:
                    latched = False
                continue

            missing = 0
            no_valid_frames = 0

            if label == last_label:
                streak += 1
            else:
                last_label = label
                streak = 1

            needed = PAPER_STREAK_NEEDED if label in PAPER_LIKE else 1
            if streak < needed:
                continue

            decision = CLASS_TO_BIN.get(label, UNKNOWN_DEFAULT_BIN)
            kind = "RECYCLABLE" if decision == "L" else "NON-RECYCLABLE"
            print(f"YOLO: {label} conf={conf:.2f} (inside ROI) => {kind} => {decision} (streak {streak}/{needed})")

            if (now - last_sent_time) < DECISION_COOLDOWN_SEC:
                continue
            if latched:
                continue

            cmd = b"L" if decision == "L" else b"R"

            latest["lastCmd"] = cmd.decode()
            update_latest("command_sent")

            ok, reply = send_with_ack(ser, cmd)

            last_sent_time = time.time()
            latched = True
            latch_time = last_sent_time

            last_label = None
            streak = 0

            # ---- update counters ONLY if ACK succeeded ----
            if ok:
                if cmd == b"L":
                    recyCount += 1
                    update_latest("ACK:L")
                else:
                    nonRecyCount += 1
                    update_latest("ACK:R")
                latest["lastAck"] = reply.decode(errors="ignore")
            else:
                latest["lastAck"] = reply.decode(errors="ignore") if reply else "NO_ACK"
                update_latest("blocked_or_failed")

            print(f">>> SENT {cmd.decode()} {'OK' if ok else 'FAILED'} | Arduino:{latest['lastAck']}")

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if picam2 is not None:
                picam2.stop()
                picam2.close()
        except Exception:
            pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
