# 필요한 라이브러리 가져오기
import cv2
from pyzbar.pyzbar import decode
import numpy as np
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
    메인 함수: pyzbar 라이브러리를 사용하여 QR 코드를 감지하고 디코딩합니다.
    """
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("--- pyzbar 기반 QR 코드 탐지 시작 ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임에서 QR 코드 찾기
        decoded_objects = decode(frame)

        found_qr = False
        data = ""

        if decoded_objects:
            for obj in decoded_objects:
                # QR 코드 경계선 그리기
                points = obj.polygon
                if len(points) > 4 : 
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    hull = np.int32(hull)
                    cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
                else :
                    cv2.polylines(frame, [np.int32(obj.polygon)], True, (0, 255, 0), 2)
                    
                # 디코딩된 데이터 출력
                data = obj.data.decode('utf-8')
                print(f"QR 인식됨: {data}")
                found_qr = True
                
                # 데이터가 있으면 루프 종료
                break
        
        # 화면에 메시지 표시
        if found_qr:
            display_frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 30), (0, 255, 0))
        else:
            display_frame = put_text_on_frame(frame, "QR 코드 찾는 중...", (10, 30), (255, 0, 0))

        # 프레임 화면에 출력
        cv2.imshow("QR 코드 리더", display_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
