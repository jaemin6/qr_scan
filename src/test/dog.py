from ultralytics import YOLO
import cv2

# 사전 학습된 YOLOv8 모델 불러오기
model = YOLO('yolov8n.pt')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 반려동물 추적을 위한 딕셔너리
track_history = {} # {track_id: [(x1, y1), (x2, y2), ...]}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 반려동물(cat, dog)을 탐지
    results = model.track(frame, persist=True, classes=[15, 16]) # 15: cat, 16: dog

    # 감지된 객체 정보 가져오기
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            # 각 반려동물의 위치(중심 좌표) 저장
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((x, y))

            # 화면에 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x-w/2), int(y-h/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('YOLOv8 Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()