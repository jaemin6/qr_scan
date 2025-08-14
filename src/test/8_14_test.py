"""
1. 캠에서 qr 코드를 인식하고, 인식된 qr 코드가 URL인 경우 해당 URL을 브라우저로 이동 할 수 있게하는 코드
2. 캠에서 qr을 인식하면 터미널에 인식 된 url주소가 뜸
3. 터미널에 인식된 주소를 ctrl + click하면 브라우저로 이동
4. q 키를 누르면 프로그램 종료
"""

import cv2
import webbrowser

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()
    print("QR 코드 인식 시작 ('q'로 종료)")

    last_data = None

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