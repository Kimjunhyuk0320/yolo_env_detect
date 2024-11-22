import pyzed.sl as sl
import cv2
import numpy as np

# 글로벌 변수로 Depth 데이터를 저장
depth_np = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global depth_np
    if event == cv2.EVENT_MOUSEMOVE:  # 마우스를 움직일 때
        if depth_np is not None:
            depth_value = depth_np[y, x]  # 마우스 위치의 Depth 값
            if np.isfinite(depth_value):  # 유효한 값인지 확인
                print(f"Depth at ({x}, {y}): {depth_value:.2f}m")  # 터미널 출력
            else:
                print(f"Depth at ({x}, {y}): Invalid/Out of range")  # 비유효한 Depth

def main():
    global depth_np

    # ZED 카메라 초기화
    zed = sl.Camera()

    # 초기 설정
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Depth 모드
    init_params.coordinate_units = sl.UNIT.METER  # 단위: 미터

    # 카메라 열기
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        return

    # 런타임 매개변수 생성
    runtime_params = sl.RuntimeParameters()

    # Depth 이미지를 저장할 객체 생성
    depth_image = sl.Mat()

    # OpenCV 창 생성 및 마우스 콜백 함수 연결
    cv2.namedWindow("Depth Map")
    cv2.setMouseCallback("Depth Map", mouse_callback)

    print("Move the mouse over the Depth Map to see depth values. Press 'q' to quit.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Depth 데이터 가져오기
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            # Depth 데이터를 Numpy 배열로 변환
            depth_np = depth_image.get_data()

            # Depth 데이터를 정규화하여 시각화
            depth_colormap = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

            # OpenCV 창에 Depth 데이터 표시
            cv2.imshow("Depth Map", depth_colormap)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 카메라 닫기 및 리소스 정리
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
