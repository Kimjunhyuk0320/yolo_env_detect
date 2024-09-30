import cv2
from ultralytics import YOLO

# YOLOv8 세그멘테이션 모델 로드
model = YOLO('yolov8s-seg.pt')

# 웹캠 캡처 시작 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 세그멘테이션 수행
    results = model(frame)

    # 결과 이미지 보여주기
    annotated_frame = results[0].plot()  # 결과 이미지를 시각화
    cv2.imshow('YOLOv8 Segmentation', annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
