# pro_1_compare.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_enhance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    if mean < 80:
        gamma = 0.6
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
        frame = cv2.LUT(frame, table)
    elif mean > 180:
        gamma = 1.4
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

def run_test(auto=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return 0, 0

    detector = cv2.QRCodeDetector()
    total_attempts = 0
    success_count = 0

    print(f"모드: {'밝기 보정' if auto else '원본'} / 'q' 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 모드에 따라 영상 선택
        processed = auto_enhance(frame) if auto else frame

        total_attempts += 1
        data, bbox, _ = detector.detectAndDecode(processed)

        if data:
            success_count += 1
            cv2.putText(processed, "Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(processed, "Not Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(f"{'Auto Enhance' if auto else 'Original'} Mode", processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return total_attempts, success_count

def plot_results(results):
    labels = ['원본', '밝기 보정']
    rates = [(s/t)*100 if t > 0 else 0 for t, s in results]

    plt.bar(labels, rates, color=['gray', 'green'])
    plt.ylabel("성공률 (%)")
    plt.title("QR 코드 인식률 비교")
    for i, v in enumerate(rates):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 100)
    plt.show()

if __name__ == "__main__":
    print("=== 1단계: 원본 영상 테스트 ===")
    orig_attempts, orig_success = run_test(auto=False)

    print("\n=== 2단계: 밝기 보정 영상 테스트 ===")
    auto_attempts, auto_success = run_test(auto=True)

    print("\n=== 결과 ===")
    print(f"원본 성공률: {orig_success}/{orig_attempts} ({(orig_success/orig_attempts)*100:.1f}%)")
    print(f"밝기 보정 성공률: {auto_success}/{auto_attempts} ({(auto_success/auto_attempts)*100:.1f}%)")

    plot_results([
        (orig_attempts, orig_success),
        (auto_attempts, auto_success)
    ])
