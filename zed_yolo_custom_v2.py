import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO

def get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2, step=2, max_attempts=10):
    """
    유효한 Depth 값을 탐색하여 반환.
    중심점(cx, cy)을 기준으로 주변 픽셀을 탐색하며 범위를 점차 늘림.
    탐색 범위를 바운딩 박스 내부로 제한.
    """
    h, w = depth_np.shape
    search_range = step

    for attempt in range(max_attempts):
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                nx, ny = cx + dx, cy + dy
                if x1 <= nx <= x2 and y1 <= ny <= y2 and 0 <= ny < h and 0 <= nx < w:
                    depth_value = depth_np[ny, nx]
                    if np.isfinite(depth_value):
                        return depth_value
        search_range += step

    return 0.0

def calculate_box_dimensions(x1, x2, y1, y2, depth, fx, fy):
    """
    Bounding Box의 실제 너비와 높이를 계산. 
    초점 거리를 이용해서 실제 거리를 구함
    """
    pixel_width = x2 - x1
    pixel_height = y2 - y1
    real_width = (pixel_width * depth) / fx
    real_height = (pixel_height * depth) / fy
    return real_width, real_height

def process_detection_results(results, depth_np, fx, fy, annotated_frame, model_names):
    """
    YOLO 탐지 결과를 처리하고, 거리 및 크기 정보를 표시.
    """
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중심점 계산
        class_name = model_names[int(box.cls[0])]  # 클래스 이름
        conf = float(box.conf[0])  # 신뢰도

        # 중심 픽셀의 깊이 값 가져오기
        depth_value = get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2)
        depth_text = f"Depth: {depth_value:.2f}m" if depth_value > 0 else "Depth: Invalid"

        if depth_value > 0:
            if class_name in ["rocks", "stone"]:
                real_width, real_height = calculate_box_dimensions(x1, x2, y1, y2, depth_value, fx, fy)
                width_text = f"Width: {real_width:.2f}m"
                height_text = f"Height: {real_height:.2f}m"

                # 거리, 가로, 세로 크기 텍스트 표시 (중심 좌표 기준)
                cv2.putText(annotated_frame, depth_text, (cx - 50, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(annotated_frame, width_text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, height_text, (cx - 50, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # 거리만 표시 (중심 좌표 기준)
                cv2.putText(annotated_frame, depth_text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            # 유효하지 않은 깊이일 경우
            cv2.putText(annotated_frame, depth_text, (cx - 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def initialize_zed_camera():
    """ZED 카메라 초기화 및 설정."""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        return None, None

    runtime_params = sl.RuntimeParameters()
    return zed, runtime_params

def main():
    # ZED 카메라 초기화
    zed, runtime_params = initialize_zed_camera()
    if not zed:
        return

    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy

    # YOLO 모델 로드
    model = YOLO("customtrain.pt")
    print("Press 'q' to quit.")

    image = sl.Mat()
    depth_image = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 이미지 가져오기
            zed.retrieve_image(image, sl.VIEW.LEFT)
            rgba_frame = image.get_data()
            rgb_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2RGB)

            # 깊이 데이터 가져오기
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_np = depth_image.get_data()

            # YOLO 탐지 수행
            results = model(rgb_frame)
            annotated_frame = results[0].plot()  # YOLO 기본 바운딩 박스 표시

            # 탐지 결과 추가 처리
            process_detection_results(results, depth_np, fx, fy, annotated_frame, model.names)

            # 결과 표시
            cv2.imshow("YOLO + ZED", annotated_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 리소스 정리
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
