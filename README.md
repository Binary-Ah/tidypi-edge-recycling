# TidyPi — Raspberry Pi + Coral Edge TPU 오염도 판별 자동 분리수거기

2025-2 AI Embedded(클라우드/AI 임베디드) 수업 팀 프로젝트 · 팀 진미채 (이진아 · 박채은, 2인)

## Problem
재활용품은 재질별로 분류해야 할 뿐 아니라 오염(음식물 등)되어 있으면 재활용이 불가능하다. 저가 임베디드 보드에서 **재질 × 오염 여부**를 실시간으로 판별하고, 더러운 쓰레기는 세척 후 분류하는 장치를 만들었다.

## System
```
Pi Camera ─► YOLOv8n (INT8, Edge TPU) ─► class = {Clean/Dirty} × {Plastic/Paper/Can/…}
                                            │
                       Dirty ──► 서보 덤퍼 + 워터펌프 3초 세척 ──┐
                       Clean ─────────────────────────────────┴─► 게이트 서보로 재질별 투입구 분류
```
- 추론: `tflite_runtime` + `libedgetpu` delegate, 320×320 입력, conf 0.60 / IoU 0.50, 추론 5~30 ms
- 하드웨어: Raspberry Pi, Coral USB Edge TPU, PCA9685 16ch PWM, SG90×3 + MG996R×2 서보, 워터펌프(5V 5A)
- 데이터: AI Hub 재활용품 이미지 약 67,813장 — JSON 라벨 22만 개 파서를 직접 만들어 `DIRTINESS` 메타데이터로 Dirty 클래스 매핑
- 최적화: INT8 양자화로 모델 크기 75% 감소, 카메라 초기화 지연 44초 → 해소, 프레임 큐 + 스레드로 추론·표시 분리, 최근 N프레임 다수결(`deque`+`Counter`)로 클래스 안정화

## Layout
```
src/
  detect_object.py            # Edge TPU 추론 + 다수결 판정 + 서보 제어 연동 (최종)
  re_detect.py                # detect_object 개선 버전(재탐지 로직)
  recycle_detect.py           # 단일 스레드 TFLite 추론 + NMS 구현 (초기 버전)
  run_pi.py                   # 카메라 → TFLite 추론 최소 예제
  web_camera.py               # Flask MJPEG 스트리밍으로 원격 확인
  servo_sorting_system.py     # PCA9685 서보/게이트/덤퍼 제어 (all.py)
  servo_sorting_system_v2.py  # 지연 초기화·안전 각도 추가 버전 (all_2.py)
docs/training_screenshot.png  # 학습 화면
```
모델 가중치(`best-int8_edgetpu.tflite`, 7.6 MB)와 AI Hub 데이터는 포함하지 않았다.

## How to run (Raspberry Pi)
```bash
pip install tflite-runtime opencv-python adafruit-circuitpython-pca9685 adafruit-circuitpython-motor flask
# best-int8_edgetpu.tflite 를 src/ 에 두고
python src/detect_object.py        # 카메라 + Edge TPU + 서보
python src/web_camera.py           # http://<pi-ip>:5000 에서 스트림 확인
```

## My role
2인 팀. AI Hub 라벨 파서·데이터셋 구성, YOLOv8n 학습 및 INT8/Edge TPU 변환, Pi 추론 코드와 서보 제어 통합 (`CODE_TODO.md` 에서 역할 분담 확인).

## Note
자율주행(F1Tenth) 과제와는 별개 프로젝트다. 최종보고서 PDF(`[2025-2_AIE] 최종보고서_박채은 이진아.pdf`)는 맥북에 있다.
