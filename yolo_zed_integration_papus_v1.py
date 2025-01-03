import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO('runs/segment/train2/weights/best.pt')  # 훈련된 YOLO 모델 경로

def calculate_real_size(pixel_width, pixel_height, depth, fx, fy):
    """Papus 정리를 사용하여 실제 크기 계산"""
    real_width = (pixel_width * depth) / fx  # 실제 너비 계산
    real_height = (pixel_height * depth) / fy  # 실제 높이 계산
    return real_width, real_height

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

    # 카메라 파라미터 가져오기
    calibration_params = zed.get_camera_information().calibration_parameters
    fx = calibration_params.left_cam.fx  # 초점 거리 (가로)
    fy = calibration_params.left_cam.fy  # 초점 거리 (세로)

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
                    object_class = box.cls  # 탐지된 객체 클래스 (예: stone, sand 등)

                    # 객체의 중심 좌표 계산
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Bounding Box 내 평균 Depth 계산
                    depth_values = []
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            depth_value = depth_np[y, x]
                            if np.isfinite(depth_value) and depth_value > 0:
                                depth_values.append(depth_value)

                    if depth_values:
                        average_depth = sum(depth_values) / len(depth_values)
                        depth_text = f"Depth: {average_depth:.2f}m"
                    else:
                        depth_text = "Depth: Invalid"

                    # 특정 클래스에 따른 처리
                    if object_class == 'stone':
                        # Papus 정리를 사용한 실제 크기 계산
                        pixel_width = x2 - x1
                        pixel_height = y2 - y1
                        if depth_values:  # 평균 Depth가 유효할 경우
                            real_width, real_height = calculate_real_size(pixel_width, pixel_height, average_depth, fx, fy)
                            real_size_text = f"Size: {real_width:.2f}m x {real_height:.2f}m"
                        else:
                            real_size_text = "Size: N/A"

                        # 결과 텍스트 표시
                        cv2.putText(result_frame, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(result_frame, real_size_text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    else:
                        # stone 외 객체는 Depth 정보만 표시
                        cv2.putText(result_frame, f"{label}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(result_frame, depth_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # 경계 상자 그리기
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
