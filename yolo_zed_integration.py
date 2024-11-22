import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO('runs/segment/train2/weights/best.pt')  # 훈련된 YOLO 모델 경로

def main():
    # ZED 카메라 초기화
    zed = sl.Camera()

    # 초기 설정
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Depth 모드
    init_params.coordinate_units = sl.UNIT.METER  # 단위: 미터
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 해상도 설정

    # 카메라 열기
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        return

    # 런타임 매개변수 생성
    runtime_params = sl.RuntimeParameters()

    # 이미지와 Depth 데이터를 저장할 객체 생성
    image = sl.Mat()
    depth_image = sl.Mat()

    print("Press 'q' to quit.")

    while True:
        # ZED 카메라 데이터 가져오기
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # RGB 이미지 가져오기
            zed.retrieve_image(image, sl.VIEW.LEFT)
            rgb_frame = image.get_data()

            # YOLO 모델로 객체 탐지 수행
            results = model(rgb_frame)  # YOLO 모델로 탐지 수행
            result_frame = rgb_frame.copy()  # 결과를 표시할 프레임 복사

            # Depth 데이터 가져오기
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_np = depth_image.get_data()

            # 탐지 결과 처리
            for box in results[0].boxes:
                if box.conf > 0.35:  # 신뢰도 임계값
                    # 경계 상자 정보 가져오기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표 변환
                    label = f"{box.cls}: {box.conf:.2f}"

                    # 객체의 중심 좌표 계산
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # 중심 좌표의 Depth 값 가져오기
                    depth_value = depth_np[center_y, center_x]

                    # 유효한 Depth 값 확인
                    if np.isfinite(depth_value):
                        depth_text = f"{depth_value:.2f}m"
                    else:
                        depth_text = "Invalid"

                    # 경계 상자와 텍스트 표시
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 경계 상자
                    cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(result_frame, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # OpenCV 창에 결과 표시
            cv2.imshow("YOLO + ZED", result_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 카메라 닫기 및 리소스 정리
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
