import numpy as np
import pandas as pd
from ultralytics import YOLO

def evaluate_model_and_save_to_csv(model_path, data_yaml):
   
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 검증 데이터셋 평가
    results = model.val(data=data_yaml)

   

# 모델 경로와 데이터셋 경로 설정
model_path = "customtrain.pt"  # YOLO 모델 경로
data_yaml = "yolo_env_detection_ver3-4/data.yaml"  # 데이터셋 YAML 경로

# 검증 실행 및 결과 저장
evaluate_model_and_save_to_csv(model_path, data_yaml)
