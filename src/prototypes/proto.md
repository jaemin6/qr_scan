# 프로토 타입 코드

# pro_1.py
# Usage: python pro_1.py  (Press 'q' to quit)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_enhance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    # Simple heuristics:
    # - Too dark: brighten with gamma < 1
    # - Too bright/washed: darken with gamma > 1
    if mean < 80:
        gamma = 0.6  # brighten
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
    elif mean > 180:
        gamma = 1.4  # darken
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
    else:
        # Mild local contrast boost (CLAHE) in normal lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found. Try a different index (e.g., 1) or check permissions.")
        return

    detector = cv2.QRCodeDetector()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = auto_enhance(frame)

        # Try decode on enhanced first, fallback to raw
        data, bbox, _ = detector.detectAndDecode(enhanced)
        if not data:
            data, bbox, _ = detector.detectAndDecode(frame)

        # Draw simple UI
        preview = enhanced.copy()
        h, w = preview.shape[:2]
        msg = "QR: " + (data if data else "—")
        cv2.putText(preview, msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(preview, msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        if bbox is not None and len(bbox) > 0:
            pts = bbox.astype(int).reshape(-1,2)
            for i in range(len(pts)):
                cv2.line(preview, tuple(pts[i]), tuple(pts[(i+1)%len(pts)]), (0,255,0), 2)

        cv2.imshow("QR HelloWorld", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# pro_1.py