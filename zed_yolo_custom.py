import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO

def get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2, step=2, max_attempts=10):
    """
    유효한 Depth 값을 탐색하여 반환.
    중심점(cx, cy)을 기준으로 주변 픽셀을 탐색하며 범위를 점차 늘림.
    탐색 범위를 바운딩 박스 내부로 제한.

    Args:
    - depth_np: Depth 데이터 배열.
    - cx, cy: 중심점 좌표.
    - x1, x2, y1, y2: 바운딩 박스의 좌우 및 위아래 경계.
    - step: 탐색 범위를 늘리는 단계 (픽셀 단위).
    - max_attempts: 최대 탐색 시도 횟수.

    Returns:
    - 유효한 Depth 값(float). 유효한 값이 없으면 0.0 반환.
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
        search_range += step  # 탐색 범위 확장

    return 0.0

def calculate_box_width(x1, x2, cx, cy, y1, y2, depth_np):
    """
    바운딩 박스의 가로 길이를 파푸스 중선정리를 이용하여 계산.

    Args:
    - x1, x2: 바운딩 박스의 좌우 x 좌표.
    - cx, cy: 바운딩 박스 중심점 좌표.
    - y1, y2: 바운딩 박스의 위쪽과 아래쪽 y 좌표.
    - depth_np: Depth 데이터 배열.

    Returns:
    - 가로 길이(float).
    """
    depth_left = get_valid_depth_in_bbox(depth_np, x1, cy, x1, x2, y1, y2)
    depth_right = get_valid_depth_in_bbox(depth_np, x2, cy, x1, x2, y1, y2)
    depth_center = get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2)

    if depth_left == 0.0 or depth_right == 0.0 or depth_center == 0.0:
        return 0.0  # 유효하지 않은 경우 0 반환

    box_width = 2 * np.sqrt((depth_left**2 + depth_right**2) / 2 - depth_center**2)
    return box_width

def calculate_box_height(y1, y2, cx, cy, x1, x2, depth_np):
    """
    바운딩 박스의 세로 길이를 파푸스 중선정리를 이용하여 계산.

    Args:
    - y1, y2: 바운딩 박스의 위쪽과 아래쪽 y 좌표.
    - cx, cy: 바운딩 박스 중심점 좌표.
    - x1, x2: 바운딩 박스의 좌우 x 좌표.
    - depth_np: Depth 데이터 배열.

    Returns:
    - 세로 길이(float).
    """
    depth_top = get_valid_depth_in_bbox(depth_np, cx, y1, x1, x2, y1, y2)
    depth_bottom = get_valid_depth_in_bbox(depth_np, cx, y2, x1, x2, y1, y2)
    depth_center = get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2)

    if depth_top == 0.0 or depth_bottom == 0.0 or depth_center == 0.0:
        return 0.0  # 유효하지 않은 경우 0 반환

    box_height = 2 * np.sqrt((depth_top**2 + depth_bottom**2) / 2 - depth_center**2)
    return box_height

def calculate_box_area(x1, x2, y1, y2, cx, cy, depth_np):
    """
    바운딩 박스의 넓이를 가로와 세로 길이를 이용해 계산.

    Args:
    - x1, x2, y1, y2: 바운딩 박스의 좌표.
    - cx, cy: 바운딩 박스 중심점 좌표.
    - depth_np: Depth 데이터 배열.

    Returns:
    - 넓이(float).
    """
    box_width = calculate_box_width(x1, x2, cx, cy, y1, y2, depth_np)
    box_height = calculate_box_height(y1, y2, cx, cy, x1, x2, depth_np)

    return box_width * box_height

def process_detection_results(results, depth_np, annotated_frame, model_names):
    """
    YOLO 탐지 결과를 처리하고, 거리 및 추가 정보를 표시.

    Args:
    - results: YOLO 탐지 결과 객체.
    - depth_np: Depth 데이터 배열.
    - annotated_frame: YOLO 탐지 결과를 시각화한 프레임.
    - model_names: 클래스 이름 리스트.

    Returns:
    - 처리된 annotated_frame.
    """
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 중심점 계산
        class_name = model_names[int(box.cls[0])]  # 클래스 이름

        # Depth 값 가져오기
        depth_value = depth_np[cy, cx]
        if not np.isfinite(depth_value):  # 유효하지 않은 경우 주변 탐색
            depth_value = get_valid_depth_in_bbox(depth_np, cx, cy, x1, x2, y1, y2)

        # 거리값 텍스트 생성
        depth_text = f"Distance: {depth_value:.2f}m"

        # 클래스가 rocks, stone일 경우 가로, 세로, 넓이 추가
        additional_text1 = None
        additional_text2 = None
        additional_text3 = None
        if class_name in ["rocks", "stone", "cement"]:
            box_width = calculate_box_width(x1, x2, cx, cy, y1, y2, depth_np)
            box_height = calculate_box_height(y1, y2, cx, cy, x1, x2, depth_np)
            box_area = calculate_box_area(x1, x2, y1, y2, cx, cy, depth_np)
            additional_text1 = f"Width: {box_width:.2f}m"
            additional_text2 = f"Height: {box_height:.2f}m"
            additional_text3 = f"Area: {box_area:.2f}m^2"
        # 바운딩 박스 중심에 텍스트 표시
        text_position_x = cx - 70  # 바운딩 박스 중심 x 좌표
        text_position_y = cy  # 바운딩 박스 중심 y 좌표

        # 거리값 텍스트 표시
        cv2.putText(annotated_frame, depth_text, (text_position_x, text_position_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 추가 정보 텍스트 표시 (한 줄씩 아래로 배치)
        if additional_text1:
            cv2.putText(annotated_frame, additional_text1, (text_position_x, text_position_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, additional_text2, (text_position_x, text_position_y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, additional_text3, (text_position_x, text_position_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return annotated_frame

def initialize_zed_camera():
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
    zed, runtime_params = initialize_zed_camera()
    if not zed:
        return

    model = YOLO("customtrain.pt")
    print("Press 'q' to quit.")

    image = sl.Mat()
    depth_image = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            rgba_frame = image.get_data()

            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_np = depth_image.get_data()

            rgb_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2RGB)

            results = model(rgb_frame)

            annotated_frame = results[0].plot()
            annotated_frame = process_detection_results(results, depth_np, annotated_frame, model.names)

            cv2.imshow("ZED 2.0i + YOLO + RGB + Depth Overlay", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
