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
# 시스템에 설치된 폰트를 찾아 사용하거나, 경로를 직접 지정합니다.
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

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
    matplotlib을 사용하여 성능 그래프를 생성하고, OpenCV 이미지로 변환하여 반환합니다.
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)
    
    # 인식률 그래프
    labels_rate = ['기본', '개선']
    rates = [baseline_rate, improved_rate]
    bars1 = ax1.bar(labels_rate, rates, color=['#ff9999', '#66b3ff'])
    ax1.set_title('인식 성공률 (%)')
    ax1.set_ylim(0, 100)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
    
    # FPS 그래프
    labels_fps = ['기본', '개선']
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
        
        # FPS 및 인식률 계산
        elapsed_base = time.time() - start_time_base
        fps_base = frame_count_base / elapsed_base if elapsed_base > 0 else 0
        success_rate_base = success_count_base / frame_count_base * 100 if frame_count_base > 0 else 0

        # 그래프 생성
        graph_img = create_graph_image(success_rate_base, 0, fps_base, 0)
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame.copy(), "기본 방법 테스트 중", (10, 30), (255, 0, 0))
        if data:
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0))
        
        # 프레임과 그래프를 병합
        graph_height, graph_width, _ = graph_img.shape
        frame_height, frame_width, _ = display_frame.shape
        combined_frame = np.zeros((max(frame_height, graph_height), frame_width + graph_width, 3), dtype=np.uint8)
        combined_frame[:frame_height, :frame_width] = display_frame
        combined_frame[:graph_height, frame_width:] = graph_img
        
        cv2.imshow("QR 성능 테스트", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    # 2단계: 개선된 인식 방법 성능 측정
    print("--- 2단계: 개선된 인식 방법 성능 측정 시작 (10초) ---")
    start_time_improved = time.time()
    frame_count_improved = 0
    success_count_improved = 0
    
    while time.time() - start_time_improved < 10:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count_improved += 1

        # 개선된 인식 방법: 전처리 후 인식
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data, _, _ = detector.detectAndDecode(binarized)
        
        if data:
            success_count_improved += 1
        
        # FPS 및 인식률 계산
        elapsed_improved = time.time() - start_time_improved
        fps_improved = frame_count_improved / elapsed_improved if elapsed_improved > 0 else 0
        success_rate_improved = success_count_improved / frame_count_improved * 100 if frame_count_improved > 0 else 0

        # 그래프 생성
        graph_img = create_graph_image(success_rate_base, success_rate_improved, fps_base, fps_improved)

        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame.copy(), "개선된 방법 테스트 중", (10, 30), (0, 255, 0))
        if data:
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0))
        
        # 프레임과 그래프를 병합
        graph_height, graph_width, _ = graph_img.shape
        frame_height, frame_width, _ = display_frame.shape
        combined_frame = np.zeros((max(frame_height, graph_height), frame_width + graph_width, 3), dtype=np.uint8)
        combined_frame[:frame_height, :frame_width] = display_frame
        combined_frame[:graph_height, frame_width:] = graph_img
        
        cv2.imshow("QR 성능 테스트", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    # 최종 결과 표시
    final_graph = create_graph_image(success_rate_base, success_rate_improved, fps_base, fps_improved)
    final_frame = put_text_on_frame(np.zeros_like(frame), "테스트 완료! 'q'를 눌러 종료하세요.", (50, 50), (255, 255, 255))

    # 최종 프레임과 그래프를 병합
    graph_height, graph_width, _ = final_graph.shape
    frame_height, frame_width, _ = final_frame.shape
    combined_final = np.zeros((max(frame_height, graph_height), frame_width + graph_width, 3), dtype=np.uint8)
    combined_final[:frame_height, :frame_width] = final_frame
    combined_final[:graph_height, frame_width:] = final_graph

    cv2.imshow("QR 성능 테스트", combined_final)

    print("\n--- 분석 완료 ---")
    print(f"기본 방법: 평균 FPS {fps_base:.2f}, 인식률 {success_rate_base:.2f}%")
    print(f"개선 방법: 평균 FPS {fps_improved:.2f}, 인식률 {success_rate_improved:.2f}%")
    
    cv2.waitKey(0) # 키 입력 대기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()