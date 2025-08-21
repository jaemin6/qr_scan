# 필요한 라이브러리 가져오기
import cv2                   # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
from ultralytics import YOLO # YOLO 모델을 쉽게 사용하기 위한 라이브러리
import numpy as np           # 배열 및 행렬 연산을 위한 라이브러리
from PIL import ImageFont, ImageDraw, Image # 한글 폰트를 이미지에 그리기 위해

# OpenCV용 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(fontpath, 20)
    is_font_loaded = True
except IOError:
    is_font_loaded = False
    font = None

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def main():
    """YOLOv11 모델을 사용하여 QR 코드를 감지하고 디코딩"""
    # 1. YOLO 모델 로드
    try:
        model = YOLO('./yolov11n.pt')  # 모델 파일 경로 확인
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        print("로보플로우에서 .pt 모델 파일을 다운로드하여 이 스크립트와 같은 폴더에 저장했는지 확인하세요.")
        return

    # 2. 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 3. OpenCV QR 코드 디코더 생성
    detector = cv2.QRCodeDetector()
    print("--- YOLOv11 기반 QR 코드 탐지 시작 ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. YOLO 모델로 QR 코드 탐지
        results = model.predict(frame, conf=0.5, verbose=False)
        data = None

        # 5. 탐지된 객체 처리
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # 바운딩 박스 정보 추출 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 관심 영역(ROI) 설정
                roi = frame[y1:y2, x1:x2]

                # ROI가 유효한지 확인
                if roi.size == 0:
                    continue

                # 6. ROI에서 QR 코드 디코딩
                data, _, _ = detector.detectAndDecode(roi)

                if data:
                    # 감지된 QR 코드에 사각형 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break  # QR 코드를 찾았으므로 루프 중단

        # 7. 프레임에 메시지 표시
        if data:
            display_frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 30), (0, 255, 0))
        else:
            display_frame = put_text_on_frame(frame, "QR 코드 찾는 중...", (10, 30), (255, 0, 0))

        # 8. 화면 출력
        cv2.imshow("YOLO QR 코드 감지", display_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
