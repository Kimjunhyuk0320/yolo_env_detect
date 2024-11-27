import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

model = YOLO('runs/segment/train2/weights/best.pt')

def get_3d_point(zed, x, y):
    point_cloud = sl.Mat()
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    point3D = point_cloud.get_value(x, y)
    return sl.Vector3(point3D[0], point3D[1], point3D[2])

def calculate_real_size(zed, x1, y1, x2, y2):
    point1 = get_3d_point(zed, x1, y1)
    point2 = get_3d_point(zed, x2, y2)
    width = abs(point2.x - point1.x)
    height = abs(point2.y - point1.y)
    return width, height

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        return

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            results = model(frame)
            result_frame = frame.copy()

            for box in results[0].boxes:
                if box.conf > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{box.cls}: {box.conf:.2f}"

                    real_width, real_height = calculate_real_size(zed, x1, y1, x2, y2)
                    size_text = f"Size: {real_width:.2f}m x {real_height:.2f}m"

                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(result_frame, size_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow("YOLO + ZED", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()