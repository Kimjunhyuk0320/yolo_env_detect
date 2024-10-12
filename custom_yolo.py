import cv2
import os
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.font_manager as fm


# YOLOv8 세그멘테이션 모델 불러오기
model = YOLO('runs/segment/train/weights/best.pt')  # 훈련된 모델 경로로 수정

def detect_objects(image_path, conf_threshold=0.35):
    # 이미지를 불러오기
    img = cv2.imread(image_path)

    # 객체 탐지 및 세그멘테이션 수행
    results = model(img)

    # 결과 시각화 (탐지된 객체 이미지 얻기)
    result_img = img.copy()  # 원본 이미지를 복사하여 사용

    # 탐지된 객체 중 신뢰도가 conf_threshold 이상인 경우만 필터링
    for box in results[0].boxes:
        if box.conf > conf_threshold:
            # 신뢰도가 높은 객체에 대해 결과를 그리기
            result_img = results[0].plot()

    return result_img  # 예측 결과 이미지 반환

def visualize_and_save_results(image_paths, save_path):
    # 한 화면에 10개의 예측 결과를 가로로 시각화하고 저장
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2행 5열로 배치 (예측 결과)
    axes = axes.ravel()

    # 예측 결과 표시
    for idx, image_path in enumerate(image_paths):
        result_img = detect_objects(image_path)  # 예측 결과
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(result_img_rgb)
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # 제목이 잘리거나 겹치지 않도록 레이아웃 조정
    
    # 저장할 이미지 경로 지정
    fig.savefig(save_path, bbox_inches='tight')

    # 결과를 화면에 표시
    plt.show()

    plt.close(fig)

if __name__ == "__main__":
    # 이미지가 저장된 폴더 경로 지정
    image_folder = 'img'  # 탐지할 이미지 폴더로 수정
    save_folder = 'result'  # 결과 저장 폴더

    # 저장 폴더가 없으면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 현재 날짜와 시간 가져오기
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일 이름에 날짜와 시간을 추가하여 저장할 파일 경로 생성
    save_path = os.path.join(save_folder, f'combined_result_{current_time}.png')

    # img 폴더 내에서 랜덤으로 10개의 이미지 선택
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(image_files, 10)

    # 선택된 10개의 이미지들에 대해 객체 탐지 결과를 나란히 시각화하고 저장
    visualize_and_save_results(selected_images, save_path)
