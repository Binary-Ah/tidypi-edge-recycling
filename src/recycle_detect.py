import cv2
import numpy as np
import time
import sys
# 라즈베리 파이 환경에서는 tflite_runtime을 사용합니다
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==========================================
# 1. 설정 및 상수 정의
# ==========================================
MODEL_PATH = "best-int8.tflite"  # 변환한 모델 파일명
LABEL_MAP = {0: 'Plastic', 1: 'Paper', 2: 'Can', 3: 'Trash'} # 클래스 정의

INPUT_SIZE = 320    # 학습 시 사용한 이미지 크기 (예: 416 or 320)
CONF_THRESH = 0.4     # 탐지 신뢰도 임계값 (이 값 이상만 표시)
IOU_THRESH = 0.45     # NMS를 위한 IoU 임계값

# ==========================================
# 2. 유틸리티 함수 (IoU, NMS) - 강의자료 11_Tflite.pdf 참고
# ==========================================
def iou(box1, box2):
    """두 박스 간의 교집합 비율(IoU) 계산"""
    # box = [ymin, xmin, ymax, xmax]
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    intersection_area = max(0, y2 - y1) * max(0, x2 - x1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

def nms(boxes, scores, classes, iou_threshold=0.5):
    """Non-Maximum Suppression: 겹치는 박스 중 가장 신뢰도 높은 것만 남김"""
    sorted_indices = np.argsort(scores)[::-1] # 점수 높은 순 정렬
    keep_boxes = []

    while sorted_indices.size > 0:
        # 가장 점수가 높은 박스 선택
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # 선택된 박스와 나머지 박스들의 IoU 계산
        ious = np.array([iou(boxes[box_id], boxes[i]) for i in sorted_indices[1:]])

        # IoU가 임계값보다 큰(너무 많이 겹치는) 박스들은 제거
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return boxes[keep_boxes], scores[keep_boxes], classes[keep_boxes]

def filter_boxes(output_data, threshold=0.4):
    """모델 출력에서 유효한 박스만 필터링"""
    # YOLOv5 TFLite 출력 형태는 보통 [1, 6300, 9] (xywh + obj_conf + 4 classes) 등일 수 있음
    # 여기서는 일반적인 처리를 위해 output_data 구조에 맞춰 파싱이 필요함.
    # 아래 코드는 [ymin, xmin, ymax, xmax, score, class] 형태라고 가정하고 작성됨.
    # 실제 모델 출력에 따라 인덱싱 수정이 필요할 수 있습니다.
    
    # 예시: 간단한 형태의 후처리 (강의 자료 기반 변형)
    valid_boxes = []
    valid_scores = []
    valid_classes = []

    # 모델 출력의 shape을 확인하여 반복문 처리 (output_data[0]이 detections)
    for detection in output_data:
        # detection 구조: [x, y, w, h, obj_conf, class_probs...] 라고 가정
        score = detection[4] 
        if score < threshold:
            continue
        
        # 클래스 확률 중 가장 높은 것 찾기
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]
        
        # 최종 점수 = 객체 확률 * 클래스 확률
        final_score = score * class_score
        if final_score < threshold:
            continue

        # 좌표 변환 (center x, center y, w, h) -> (ymin, xmin, ymax, xmax)
        cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]
        ymin = cy - h / 2
        xmin = cx - w / 2
        ymax = cy + h / 2
        xmax = cx + w / 2

        valid_boxes.append([ymin, xmin, ymax, xmax])
        valid_scores.append(final_score)
        valid_classes.append(class_id)

    if not valid_boxes:
        return np.array([]), np.array([]), np.array([])

    return nms(np.array(valid_boxes), np.array(valid_scores), np.array(valid_classes), IOU_THRESH)

# ==========================================
# 3. 메인 실행 함수
# ==========================================
def main():
    # --- 모델 로드 (Edge TPU Delegate 포함) ---
    try:
        # Edge TPU 가속을 위해 Delegate 로드 
        delegate = load_delegate('libedgetpu.so.1')
        interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
        print("Edge TPU Delegate 로드 성공!")
    except Exception as e:
        print(f"Edge TPU 로드 실패 ({e}), CPU로 실행합니다.")
        interpreter = Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    # 입력/출력 텐서 정보 획득
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_idx = input_details[0]['index']
    output_idx = output_details[0]['index']

    # 카메라 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("카메라를 시작합니다. 종료하려면 'q'를 누르세요.")

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. 전처리 (Preprocessing) ---
        # 이미지 크기 조정 및 정규화
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        input_data = np.expand_dims(img, axis=0)

        # INT8 양자화 모델인 경우 형변환 및 스케일링 필요 [cite: 1092, 1093, 1109]
        if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            # float -> int8/uint8 변환 (Input Quantization)
            input_data = (input_data.astype(np.float32) / 255.0 / scale + zero_point).astype(input_details[0]['dtype'])
        else:
            # Float 모델인 경우
            input_data = (input_data.astype(np.float32) / 255.0)

        # --- 2. 추론 (Inference) ---
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke() # [cite: 1122]

        # --- 3. 후처리 (Post-processing) ---
        outputs = interpreter.get_tensor(output_idx)[0] # 첫 번째 배치의 결과

        # 출력 데이터 역양자화 (Dequantization)가 필요한 경우 [cite: 1141]
        # YOLO 출력은 보통 복잡하므로 모델 구조에 따라 이 부분은 생략되거나 수정될 수 있습니다.
        # 여기서는 boxes, scores, classes를 추출하여 화면에 그립니다.
        
        # (주의: YOLOv5 모델의 출력 텐서 구조에 맞춰 filter_boxes 함수 내부를 조정해야 할 수 있습니다.)
        boxes, scores, classes = filter_boxes(outputs, CONF_THRESH)

        # --- 4. 시각화 (Visualization) ---
        h, w, _ = frame.shape
        for box, score, cls in zip(boxes, scores, classes):
            ymin, xmin, ymax, xmax = box
            
            # 좌표를 원본 프레임 크기로 변환
            x1 = int(xmin * w / INPUT_SIZE)
            y1 = int(ymin * h / INPUT_SIZE)
            x2 = int(xmax * w / INPUT_SIZE)
            y2 = int(ymax * h / INPUT_SIZE)

            # 박스 그리기
            label_name = LABEL_MAP.get(int(cls), 'Unknown')
            label = f"{label_name}: {score:.2f}"
            color = (0, 255, 0) # 초록색

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS 출력
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Recycle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()