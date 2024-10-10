
# YOLOv8 Custom Training and Object Detection

## 프로젝트 개요
이 프로젝트는 Roboflow에서 데이터를 다운로드하고 YOLOv8 세그멘테이션 모델을 훈련한 후, 사용자 정의 YOLOv8 모델을 사용하여 이미지에서 객체를 탐지하는 과정을 다룹니다.

### 주요 파일
1. `train_yolo_seg.py`: Roboflow에서 데이터를 다운로드하고, YOLOv8 세그멘테이션 모델을 훈련하는 스크립트입니다.
2. `custom_yolo.py`: 훈련된 모델을 사용하여 이미지에서 객체를 탐지하고, 그 결과를 시각화 및 저장하는 스크립트입니다.

---

## 설치 및 환경 설정

### 1. Python 환경 설정

먼저, Python 3.8 이상 버전이 설치되어 있어야 합니다.

### 2. 필요한 라이브러리 설치

다음 명령어로 필요한 라이브러리를 설치합니다:

```bash
pip install -r opencv-python matplotlib roboflow ultralytics
```

---

## 프로젝트 실행

### 1. 데이터셋 다운로드 및 모델 훈련

`train_yolo_seg.py` 스크립트는 Roboflow에서 YOLOv8 형식의 데이터셋을 다운로드하고 YOLOv8 세그멘테이션 모델을 훈련합니다. 다음 명령어로 훈련을 시작하세요:

```bash
python train_yolo_seg.py
```

이 스크립트는:
- Roboflow에서 프로젝트 데이터를 다운로드합니다.
- YOLOv8 세그멘테이션 모델을 `yolov8s-seg.pt`을 기반으로 훈련합니다.
- 훈련된 모델과 데이터를 `runs/segment/train/weights/best.pt`에 저장합니다.

### 2. 이미지에서 객체 탐지

`custom_yolo.py` 스크립트를 실행하면 지정된 폴더 내의 이미지를 대상으로 객체 탐지를 수행하고 결과를 시각화합니다.

```bash
python custom_yolo.py
```

#### 스크립트 동작:
- `img/` 폴더에서 무작위로 10개의 이미지를 선택합니다.
- 선택된 이미지에서 훈련된 YOLOv8 모델을 사용하여 객체를 탐지합니다.
- 탐지된 결과를 시각화하여 `result/` 폴더에 저장하고 화면에 표시합니다.

---

## 폴더 구조

```
.
├── img/                     # 탐지할 이미지가 저장되는 폴더
├── result/                  # 결과 이미지가 저장되는 폴더
├── train_yolo_seg.py        # YOLOv8 훈련 스크립트
├── custom_yolo.py           # YOLOv8 탐지 및 시각화 스크립트
├── requirements.txt         # 설치할 외부 라이브러리 목록
└── README.md                # 프로젝트 설명 파일
```

---

## 주의사항

1. Roboflow API 키를 발급받은 후 `train_yolo_seg.py`에 입력해야 합니다.
2. YOLOv8 모델을 훈련할 때 시간이 많이 걸릴 수 있으며, GPU를 사용하는 것이 권장됩니다.
3. `img/` 폴더에 탐지할 이미지 파일을 `.png`, `.jpg`, `.jpeg` 형식으로 저장하세요.
