import pyzed.sl as sl
import cv2
import numpy as np

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
        # 카메라 데이터 가져오기
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # RGB 이미지 가져오기
            zed.retrieve_image(image, sl.VIEW.LEFT)
            rgb_frame = image.get_data()

            # Depth 데이터 가져오기
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_np = depth_image.get_data()

            # 화면에 표시할 텍스트 추가
            h, w, _ = rgb_frame.shape

            # 여러 위치의 Depth 값을 화면에 표시 (격자 형태로 예제)
            grid_size = 5  # 표시할 Depth 값의 개수
            step_x, step_y = w // grid_size, h // grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x, y = step_x * i + step_x // 2, step_y * j + step_y // 2  # 그리드 중심 좌표
                    depth_value = depth_np[y, x]

                    if np.isfinite(depth_value):  # 유효한 Depth 값만 표시
                        text = f"{depth_value:.2f}m"
                        cv2.putText(rgb_frame, text, (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # 그리드 중심을 작은 원으로 표시
                    cv2.circle(rgb_frame, (x, y), 3, (0, 255, 0), -1)

            # OpenCV 창에 RGB 영상 표시
            cv2.imshow("RGB + Depth Overlay", rgb_frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 카메라 닫기 및 리소스 정리
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()