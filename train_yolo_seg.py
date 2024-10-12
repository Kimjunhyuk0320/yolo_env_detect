import os
from roboflow import Roboflow
from ultralytics import YOLO

def download_dataset():
    # Roboflow API 키 설정 및 데이터셋 다운로드
    rf = Roboflow(api_key="DfJJy1YtAtAu4nlKAbvs")  # Roboflow API 키 입력
    project = rf.workspace("yoloproject-rnxqb").project("yolo_env_detection_ver3")  # 프로젝트 이름 입력
    dataset = project.version(1).download("yolov8")  # YOLOv8 형식으로 데이터셋 다운로드
    
    # 경로 출력해서 확인하기
    print(f"Dataset downloaded at: {dataset.location}")
    
    # 데이터셋 경로 반환
    return os.path.join(dataset.location, 'data.yaml')

def train_model(data_yaml):
    # YOLOv8 모델 로드
    model = YOLO('yolov8s-seg.pt')  # 세그멘테이션용 모델 로드
    
    # 모델 학습
    model.train(data=data_yaml, epochs=200, imgsz=640, plots=True)

def main():
    # Roboflow에서 데이터셋 다운로드 및 경로 가져오기
    data_yaml = download_dataset()

    # YOLOv8 세그멘테이션 모델 훈련
    train_model(data_yaml)

if __name__ == '__main__':
    main()
