# pro_1_visual.py
# Usage: python pro_1_visual.py  (Press 'q' to quit)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------------------------
# 자동 밝기/대비 보정 함수
# ---------------------------
def auto_enhance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

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
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return frame

# ---------------------------
# 메인 함수
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found. Try a different index (e.g., 1) or check permissions.")
        return

    detector = cv2.QRCodeDetector()

    # 인식률 측정 변수
    total_attempts = 0
    success_count = 0
    rate_history = deque(maxlen=50)  # 최근 50개 성공률 기록

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = auto_enhance(frame)

        # QR 코드 감지
        total_attempts += 1
        data, bbox, _ = detector.detectAndDecode(enhanced)
        if not data:
            data, bbox, _ = detector.detectAndDecode(frame)

        if data:
            success_count += 1

        # 성공률 계산
        success_rate = (success_count / total_attempts) * 100
        rate_history.append(success_rate)

        # 카메라 영상 표시
        preview = enhanced.copy()
        h, w = preview.shape[:2]
        msg = "QR: " + (data if data else "—")
        cv2.putText(preview, msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(preview, msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        if bbox is not None and len(bbox) > 0:
            pts = bbox.astype(int).reshape(-1,2)
            for i in range(len(pts)):
                cv2.line(preview, tuple(pts[i]), tuple(pts[(i+1)%len(pts)]), (0,255,0), 2)

        # ---------------------------
        # 인식률 게이지 시각화
        # ---------------------------
        gauge_width = 200
        gauge_height = 20
        gauge_x = w - gauge_width - 10
        gauge_y = 10

        # 게이지 배경
        cv2.rectangle(preview, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), (50, 50, 50), -1)

        # 게이지 채우기
        fill_width = int((success_rate / 100) * gauge_width)
        cv2.rectangle(preview, (gauge_x, gauge_y), (gauge_x + fill_width, gauge_y + gauge_height), (0, 255, 0), -1)

        # 성공률 텍스트
        cv2.putText(preview, f"{success_rate:.1f}%", (gauge_x + 5, gauge_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # 시도/성공 횟수 표시
        cv2.putText(preview, f"Attempts: {total_attempts}  Success: {success_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("QR HelloWorld + Success Rate", preview)

        # 종료 키
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
if __name__ == "__main__":
    main()
