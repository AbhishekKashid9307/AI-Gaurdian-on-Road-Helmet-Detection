import cv2
import numpy as np
import datetime
import os
import pywhatkit
import easyocr
from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
"""
Inference-Only App with Snapshot Alerts (one alert per plate per session):
1. Sends a captioned snapshot via WhatsApp when a violation is detected.
2. Uses EasyOCR for plate text, and Tkinter GUI for display.
3. Tracks notified plates to avoid duplicate alerts in the same run.
"""

# ----- Sample Database: License Plate -> WhatsApp Contact -----
driver_contacts = {
    "MH18EQ0001": "+919763443635",
    "TN82Y8388" : "+91 7249610635"
    # add more mappings as needed
}

# Keep track of plates already alerted this session
otified_plates = set()

# ----- Initialize EasyOCR Reader -----
reader = easyocr.Reader(['en'], gpu=False)

# ----- Load Pre-trained Model & Cascades -----
helmet_model = load_model('maskmodel.h5', compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
if face_cascade.empty() or plate_cascade.empty():
    raise IOError("Failed to load Haar cascades. Verify OpenCV installation.")

# ----- OCR Helper -----
def recognize_plate_easyocr(plate_img):
    img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    if not results:
        return None
    text, _ = max(((res[1], res[2]) for res in results), key=lambda x: x[1])
    return ''.join(c for c in text if c.isalnum()).upper()

# ----- Notification Helper -----
def send_snapshot_alert(number, message, img_frame):
    # Save annotated snapshot
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"snapshot_{timestamp}.jpg"
    cv2.imwrite(fname, img_frame)
    try:
        pywhatkit.sendwhats_image(number, fname, caption=message, wait_time=15, tab_close=False)
        print(f"[INFO] Snapshot sent to {number}")
    except Exception as e:
        print(f"[ERROR] Could not send snapshot: {e}")
    # Optional cleanup
    # os.remove(fname)

# ----- Video Capture Setup -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

# ----- Tkinter GUI Setup -----
root = tk.Tk()
root.title("Helmet & Plate Detection")
label = tk.Label(root)
label.pack()

# ----- Frame Processing Callback -----
def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(100, update_frame)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        inp = cv2.resize(roi, (150, 150)) / 255.0
        pred = helmet_model.predict(np.expand_dims(inp, axis=0))[0][0]
        violation = pred >= 0.5

        color = (0, 0, 255) if violation else (0, 255, 0)
        label_text = 'NO HELMET' if violation else 'HELMET'
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if violation:
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
            offset = (0, 0)
            if len(plates) == 0:
                ry = y + h
                roi_area = gray[ry:ry+h, x:x+w]
                plates = plate_cascade.detectMultiScale(roi_area, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
                offset = (x, ry)

            if len(plates) > 0:
                bx, by, bw, bh = max(plates, key=lambda b: b[2]*b[3])
                bx += offset[0]; by += offset[1]
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 255, 0), 2)
                plate_img = frame[by:by+bh, bx:bx+bw]

                plate_text = recognize_plate_easyocr(plate_img)
                if plate_text:
                    cv2.putText(frame, plate_text, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    contact = driver_contacts.get(plate_text)
                    if contact:
                        # Only send one alert per session per plate
                        if plate_text not in otified_plates:
                            message = f"📜 Official Fine Notice 📜\n\n🚨 Violation Alert: On {ts}, your vehicle ({plate_text}) was detected without a helmet, violating road safety statutes.\n\n⚖️ Legal Implication: This constitutes a breach of traffic law provisions, warranting a fine as per jurisdictional guidelines.\n\n🔹 Issued by the Traffic Enforcement Department"

                            send_snapshot_alert(contact, message, frame)
                            otified_plates.add(plate_text)
                        else:
                            print(f"[INFO] Already alerted for plate {plate_text}")
                    else:
                        print(f"[WARN] Unknown plate: {plate_text}")
                else:
                    print("[WARN] OCR failed to read plate.")

    cv2.putText(frame, ts, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    label.imgtk = imgtk
    label.config(image=imgtk)

    root.after(100, update_frame)

# Graceful exit
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.after(0, update_frame)
root.mainloop()
