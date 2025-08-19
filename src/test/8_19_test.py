# 필요한 라이브러리 가져오기
import cv2
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import ImageFont, ImageDraw, Image

# matplotlib 백엔드를 'Agg'로 명시적으로 설정하여 이미지 변환을 안정적으로 만듦
matplotlib.use('Agg')

# 한글 폰트 설정 (matplotlib용)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

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

def create_graph_image(baseline_rate, improved_rate, fps_base, fps_improved):
    """
    matplotlib을 사용하여 최종 성능 그래프를 생성합니다.
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)
    
    # 인식률 그래프
    labels_rate = ['기본', '향상된 방법']
    rates = [baseline_rate, improved_rate]
    bars1 = ax1.bar(labels_rate, rates, color=['#ff9999', '#66b3ff'])
    ax1.set_title('인식 성공률 (%)')
    ax1.set_ylim(0, 100)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
    
    # FPS 그래프
    labels_fps = ['기본', '향상된 방법']
    fps_values = [fps_base, fps_improved]
    bars2 = ax2.bar(labels_fps, fps_values, color=['#ff9999', '#66b3ff'])
    ax2.set_title('평균 FPS')
    ax2.set_ylim(0, max(fps_values) * 1.2 if max(fps_values) > 0 else 10)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom')
    
    # 그래프를 메모리 버퍼로 변환 (안정적인 방법)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    buf = canvas.buffer_rgba()
    graph_img = np.asarray(buf, dtype=np.uint8)
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig) # 메모리 해제
    return graph_img

def adjust_lighting(frame):
    """조명 환경 개선을 위한 CLAHE 적용"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
def remove_noise(frame):
    """노이즈 제거를 위한 미디언 필터 적용"""
    return cv2.medianBlur(frame, 5)

def main():
    """
    메인 함수: 웹캠을 통해 두 가지 QR 인식 방식의 성능을 실시간으로 비교하고 시각화합니다.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()
    
    # 1단계: 기본 인식 방법 성능 측정
    print("--- 1단계: 기본 인식 방법 성능 측정 시작 (10초) ---")
    start_time_base = time.time()
    frame_count_base = 0
    success_count_base = 0
    
    while time.time() - start_time_base < 10:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count_base += 1
        
        # 기본 인식 방법: 전처리 없이 바로 인식
        data, _, _ = detector.detectAndDecode(frame)
        if data:
            success_count_base += 1
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame.copy(), "기본 방법 테스트 중", (10, 30), (255, 0, 0))
        if data:
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0))
        
        cv2.imshow("QR 성능 테스트", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyAllWindows()
    
    # 최종 FPS 및 인식률 계산
    elapsed_base = time.time() - start_time_base
    fps_base = frame_count_base / elapsed_base if elapsed_base > 0 else 0
    success_rate_base = success_count_base / frame_count_base * 100 if frame_count_base > 0 else 0
    
    # 2단계: 향상된 방법 성능 측정
    print("--- 2단계: 향상된 방법 테스트 시작 (10초) ---")
    start_time_improved = time.time()
    frame_count_improved = 0
    success_count_improved = 0
    
    while time.time() - start_time_improved < 10:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count_improved += 1
        
        data = None
        
        # 노이즈 제거 및 조명 보정 적용
        processed_frame = remove_noise(frame.copy())
        processed_frame = adjust_lighting(processed_frame)
        
        # QR 코드 디텍터로 바운딩 박스 찾아내기
        _, points, _ = detector.detectAndDecode(processed_frame)

        if points is not None and points.size > 0:
            # 바운딩 박스 그리기
            cv2.polylines(frame, points.astype(int), True, (0, 255, 255), 3)

            # 바운딩 박스 영역만 잘라서 디코딩 시도
            x, y, w, h = cv2.boundingRect(points.astype(int))
            
            # ROI(관심 영역) 좌표가 유효한지 안전하게 확인
            y_safe = max(0, y)
            x_safe = max(0, x)
            y2_safe = min(processed_frame.shape[0], y + h)
            x2_safe = min(processed_frame.shape[1], x + w)
            
            if (x2_safe - x_safe) > 0 and (y2_safe - y_safe) > 0:
                roi_to_decode = processed_frame[y_safe:y2_safe, x_safe:x2_safe]
                data, _, _ = detector.detectAndDecode(roi_to_decode)
        
        if data:
            success_count_improved += 1
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame, "향상된 방법 테스트 중", (10, 30), (0, 255, 0))
        if data:
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0))
        
        cv2.imshow("QR 성능 테스트", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()

    # 최종 FPS 및 인식률 계산
    elapsed_improved = time.time() - start_time_improved
    fps_improved = frame_count_improved / elapsed_improved if elapsed_improved > 0 else 0
    success_rate_improved = success_count_improved / frame_count_improved * 100 if frame_count_improved > 0 else 0

    # 최종 결과 표시
    final_graph = create_graph_image(success_rate_base, success_rate_improved, fps_base, fps_improved)
    cv2.imshow("QR 성능 최종 결과", final_graph)

    print("\n--- 분석 완료 ---")
    print(f"기본 방법: 평균 FPS {fps_base:.2f}, 인식률 {success_rate_base:.2f}%")
    print(f"향상된 방법: 평균 FPS {fps_improved:.2f}, 인식률 {success_rate_improved:.2f}%")
    
    cv2.waitKey(0) # 키 입력 대기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
