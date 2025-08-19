# test3.py
# QR 코드 인식: 원본(before) vs 개선(after) 비교

import cv2
import numpy as np
import time

# 개선된 전처리 함수
def enhance_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE(대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 블러 (노이즈 완화)
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)

    return enhanced

# 기본 QR 디코딩
def decode_before(frame):
    qr_detector = cv2.QRCodeDetector()
    data, points, _ = qr_detector.detectAndDecode(frame)
    return data, points

# 개선 후 QR 디코딩
def decode_after(frame):
    qr_detector = cv2.QRCodeDetector()
    enhanced = enhance_image(frame)
    data, points, _ = qr_detector.detectAndDecode(enhanced)
    return data, points

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    total_frames = 0
    detected_before = 0
    detected_after = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # 원본 인식
        data_before, pts_before = decode_before(frame)
        if data_before:
            detected_before += 1
            cv2.putText(frame, f"Before: {data_before}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if pts_before is not None:
                pts_before = np.int32(pts_before).reshape(-1, 2)
                cv2.polylines(frame, [pts_before], True, (0, 255, 0), 2)

        # 개선 후 인식
        data_after, pts_after = decode_after(frame.copy())
        if data_after:
            detected_after += 1
            cv2.putText(frame, f"After: {data_after}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if pts_after is not None and len(pts_after) > 0:
                pts_after = np.int32(pts_after).reshape(-1, 2)
                cv2.polylines(frame, [pts_after], True, (255, 0, 0), 2)

        # 결과 표시
        cv2.imshow("QR Code Detection (Before vs After)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 최종 통계 출력
    print(f"총 프레임 수: {total_frames}")
    print(f"원본 인식 성공: {detected_before} ({detected_before/total_frames*100:.2f}%)")
    print(f"개선 인식 성공: {detected_after} ({detected_after/total_frames*100:.2f}%)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
