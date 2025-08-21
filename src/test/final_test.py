# qr_webcam_basic.py
import cv2
import time
import tkinter as tk

def run_qr_detection():
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()  # OpenCV 기본 QR 코드 디텍터

    total_frames = 0
    detected_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        # QR 코드 검출
        data, bbox, _ = detector.detectAndDecode(frame)
        if data:
            detected_frames += 1
            if bbox is not None:
                bbox = bbox.astype(int)
                n = len(bbox[0])
                for i in range(n):
                    cv2.line(frame, tuple(bbox[0][i]), tuple(bbox[0][(i+1)%n]), (0, 255, 0), 2)
                cv2.putText(frame, "QR Detected", (bbox[0][0][0], bbox[0][0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("QR Detection - Press q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed_time = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()

    # 인식률 계산
    detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0

    # Tkinter로 결과 표시
    root = tk.Tk()
    root.title("QR 인식 결과")

    msg = f"""
총 프레임 수: {total_frames}
QR 검출된 프레임 수: {detected_frames}
인식률: {detection_rate:.2f} %
총 소요 시간: {elapsed_time:.2f} 초
"""

    label = tk.Label(root, text=msg, font=("맑은 고딕", 12), justify="left")
    label.pack(padx=20, pady=20)
    root.mainloop()


if __name__ == "__main__":
    run_qr_detection()
