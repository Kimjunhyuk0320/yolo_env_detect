* 파일 작성 시 : 파일 명, 경로 기록
* 패키지 설치 시, 모든 설치 패키지 기록하기
* commit 시 : 연도날짜 - 시간 - 작성자 적기 (ex. 241118_20:16_kang)

241118 기록 시작


241127 신석
yolo_zed_integration_papus_v1.py 파일 형성
- 객체의 평균 Depth 값을 표시하고, Papus 정리를 이용한 실제 크기 추정값을 표시하도록 개선된 코드


241127 신석
yolo_zed_integration_papus_v1.py 파일 형성
- 객체의 평균 Depth 값을 표시하고, Papus 정리를 이용한 실제 크기 추정값을 표시하도록 개선된 코드

---
241119 준혁
model_test.py 실행시
pip install scikit-learn

---
241121 신석, (황)준혁

[test] ZED 실행 기본 코드 작성 (아래 파일 4개)
zed_test 파일 추가 완료
1. depth_display.py
    - 파란색이 디폴트
    - 왼쪽 상단 Depth 값 하나만 표시 : 카메라 중앙 부분의 Depth만 전달
2. depth_display_sep
    - 위의 파일과 같음
    - 마우스 커서 가져다 대는 곳의 Depth를 터미널에 표기
3. depth_rgb
    - 일반 rgb 화면에서 matrix 처럼 점이 찍혀있고, 각각의 점의 depth를 표기함

ultralytics 설치