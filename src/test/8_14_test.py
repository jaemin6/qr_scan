import cv2
import webbrowser

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        if data:
            # QR 코드에서 URL 인식 했을 때
            # URL인지 확인하는
            if data.startswith("http") or data.startswith("https"):
                print(f"URL 인식됨: {data}")
                webbrowser.open(data)


        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()