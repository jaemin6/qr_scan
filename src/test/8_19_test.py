# 필요한 라이브러리 가져오기
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 한글 폰트 설정을 위한 유틸리티 함수 (matplotlib, OpenCV)
from PIL import ImageFont, ImageDraw, Image

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
    """
    메인 함수: YOLOv8 모델을 사용하여 QR 코드를 감지하고 디코딩합니다.
    """
    # YOLOv8n 모델 로드 (n: nano)
    # YOLO 모델은 COCO 데이터셋으로 학습되어 있어 'qr code'를 바로 인식하지 못합니다.
    # 따라서, 'cell phone', 'keyboard' 등 QR 코드가 자주 등장하는 다른 객체를 찾아야 합니다.
    # 더 정확한 QR 코드 인식을 위해서는 QR 코드가 포함된 사용자 정의 데이터셋으로 모델을 재학습해야 합니다.
    # 여기서는 'tv', 'cell phone', 'book' 등 비슷한 형태의 객체를 가정하고 탐지합니다.
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()
    
    # YOLO 모델 사용
    print("--- YOLOv8 기반 QR 코드 탐지 시작 ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO 모델로 객체 탐지
        # 여기서는 'tv' 클래스(클래스 번호 62)를 QR 코드라고 가정합니다.
        results = model.predict(frame, classes=[62], conf=0.5, verbose=False)
        
        data = None
        
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # 바운딩 박스 정보 추출 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 관심 영역(ROI) 설정
                yolo_roi = frame[y1:y2, x1:x2]
                
                # ROI가 유효한지 확인
                if not yolo_roi.shape[0] > 0 or not yolo_roi.shape[1] > 0:
                    continue

                # ROI에서 QR 코드 디코딩
                data, _, _ = detector.detectAndDecode(yolo_roi)
                
                if data:
                    # 감지된 QR 코드에 사각형 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break # QR 코드를 찾았으므로 루프 중단

        # 프레임에 메시지 표시
        if data:
            display_frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 30), (0, 255, 0))
        else:
            display_frame = put_text_on_frame(frame, "QR 코드 찾는 중...", (10, 30), (255, 0, 0))
        
        cv2.imshow("YOLO QR 코드 감지", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
