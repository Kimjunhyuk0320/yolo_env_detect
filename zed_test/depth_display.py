import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Configure initial parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Units in meters

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        return

    # Create runtime parameters
    runtime_params = sl.RuntimeParameters()

    # Create Mat objects to store images
    depth_image = sl.Mat()
    depth_colormap = None

    print("Press 'q' to quit.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve depth data
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            # Convert depth data to OpenCV-compatible format
            depth_np = depth_image.get_data()

            # Normalize for visualization
            depth_colormap = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

            # Add depth values as text on the image (center pixel as an example)
            h, w = depth_np.shape
            center_x, center_y = w // 2, h // 2
            depth_value = depth_np[center_y, center_x]  # Depth at center pixel
            text = f"Depth: {depth_value:.2f}m"

            # Overlay text on the colormap image
            cv2.putText(depth_colormap, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the depth map with the depth value
            cv2.imshow("Depth Map", depth_colormap)

            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
