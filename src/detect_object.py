from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from collections import deque, Counter
import cv2
import numpy as np
import os
import sys
import time

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    try:
        # 데스크탑 환경에서 tflite_runtime이 없을 때의 폴백
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
    except Exception:
        Interpreter = None
        load_delegate = None

# Don't create Interpreter at import time (EdgeTPU models require delegate).
def print_model_io(model_path):
    from tflite_runtime.interpreter import Interpreter, load_delegate
    try:
        delegate = load_delegate('libedgetpu.so.1')
        interp = Interpreter(model_path=str(model_path), experimental_delegates=[delegate])
        print("Using EdgeTPU delegate")
    except Exception:
        interp = Interpreter(model_path=str(model_path))
        print("Using CPU interpreter (no EdgeTPU)")
    interp.allocate_tensors()
    print("INPUT DETAILS:", interp.get_input_details())
    print("OUTPUT DETAILS:", interp.get_output_details())

HEADLESS = os.environ.get("DISPLAY") is None

# ==========================================
# 1. 설정 및 상수 정의 (성능 및 정확도 튜닝)
# ==========================================
MODEL_PATH = "best-int8_edgetpu.tflite" # 컴파일된 모델 파일명 (실제 파일명으로 변경 필요)
INPUT_SIZE = 320      # 컴파일에 사용한 크기 (Edge TPU 호환성을 위해 320으로 고정)

# === 바운딩 박스 난잡함 개선을 위한 임계값 조정 ===
CONF_THRESH = 0.60    # [난잡함 개선] 최소 신뢰도 임계값: 0.60 (낮은 확신 박스 필터링)
IOU_THRESH = 0.50     # [난잡함 개선] NMS IoU 임계값: 겹치는 박스를 공격적으로 합침 (0.50)

# 7개 클래스 (오염도 반영)
LABEL_MAP = {
    0: 'Clean_Plastic', 1: 'Dirty_Plastic', 2: 'Paper',
    3: 'Clean_Can', 4: 'Dirty_Can', 5: 'Bottle', 6: 'General_Trash'
}
COLOR_MAP = {
    0: (0, 255, 0),    # Clean Plastic (녹색)
    1: (0, 0, 255),    # Dirty Plastic (빨간색 - 세척 필요)
    2: (0, 255, 255),  # Paper (노란색)
    3: (255, 255, 0),  # Clean Can (청록색)
    4: (255, 0, 0),    # Dirty Can (파란색 - 세척 필요)
    5: (255, 0, 255),  # Bottle (자홍색)
    6: (128, 128, 128) # General Trash (회색)
}

# 변경: 캡처/처리 관련 상수 추가
PROCESS_EVERY_N = 2          # 기본: 프레임당 1/2 처리 (조절 가능)
MIN_PROCESS_N = 1
MAX_PROCESS_N = 6
TARGET_LATENCY_MS = 50.0     # 목표 지연 (동적 조정용)

try:
    from all_2 import control_system_loop
    ALL2_AVAILABLE = True
except Exception as e:
    ALL2_AVAILABLE = False
    print("[WARN] all_2 import failed - hardware trigger disabled.", e)

TRIGGER_COOLDOWN_SEC = 10.0
VOTE_WINDOW = 5
VOTE_REQUIRED = 3
PERSIST_FRAMES = 6
trigger_last_time = 0.0
trigger_running = False

# ==========================================
# 2. 유틸리티 함수 (IoU, NMS)
# ==========================================
def iou(box1, box2):
    """두 박스 간의 교집합 비율(IoU) 계산 — 입력을 float으로 처리하여 overflow 방지"""
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    intersection_area = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0

def nms(boxes, scores, classes, iou_t=IOU_THRESH, max_output=50):
    """클래스별 NMS -> 중복 제거 + 결과 수 제한"""
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    keep_indices = []
    scale = 1000.0  # cv2 expects ints; scale normalized coords
    for cls in np.unique(classes):
        inds = np.where(classes == cls)[0]
        if inds.size == 0:
            continue
        rects = []
        score_list = []
        for i in inds:
            ymin, xmin, ymax, xmax = boxes[i]
            x = int(xmin * scale)
            y = int(ymin * scale)
            w = max(1, int((xmax - xmin) * scale))
            h = max(1, int((ymax - ymin) * scale))
            rects.append([x, y, w, h])
            score_list.append(float(scores[i]))
        idxs = cv2.dnn.NMSBoxes(rects, score_list, float(CONF_THRESH), float(iou_t))
        if len(idxs) == 0:
            continue
        # flatten nested structure
        for entry in idxs:
            pick = int(entry[0]) if hasattr(entry, "__len__") else int(entry)
            keep_indices.append(inds[pick])

    if not keep_indices:
        return np.array([]), np.array([]), np.array([])

    keep_indices = np.array(keep_indices, dtype=np.int32)
    # 정렬하고 개수 제한
    order = np.argsort(scores[keep_indices])[::-1]
    keep_indices = keep_indices[order][:max_output]

    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]


def filter_boxes(output_data, threshold=CONF_THRESH, max_results=50):
    """
    일반적인 YOLO-like 출력을 다루도록 개선.
    기대 형식 예: [cx, cy, w, h, obj_conf, class_probs...]
    또는 [ymin, xmin, ymax, xmax, score, class_id]
    """
    if output_data is None:
        return np.array([]), np.array([]), np.array([])

    arr = np.array(output_data, dtype=np.float32)
    if arr.size == 0:
        return np.array([]), np.array([]), np.array([])

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    # case A: xywh + objectness + class_probs
    if arr.shape[1] >= 6:
        boxes_raw = arr[:, :4]
        obj_conf = arr[:, 4]
        class_probs = arr[:, 5:]

        if class_probs.size == 0:
            classes = np.zeros(obj_conf.shape[0], dtype=np.int32)
            class_conf = np.ones_like(obj_conf)
        else:
            class_id = np.argmax(class_probs, axis=1)
            class_conf = np.max(class_probs, axis=1)
            classes = class_id.astype(np.int32)

        scores = obj_conf * class_conf

        # 좌표 포맷 판단: 값 범위가 0..1 이면 cx,cy,w,h (정규화)로 처리
        if boxes_raw.max() <= 1.0 + 1e-6:
            cx = boxes_raw[:, 0]; cy = boxes_raw[:, 1]; w = boxes_raw[:, 2]; h = boxes_raw[:, 3]
            ymin = cy - h / 2.0
            xmin = cx - w / 2.0
            ymax = cy + h / 2.0
            xmax = cx + w / 2.0
            boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)
        else:
            # 픽셀 좌표일 가능성: 정규화 시도 (정규화 정보가 변동적이면 디버깅 필요)
            maxv = boxes_raw.max()
            if arr.shape[1] >= 6 and boxes_raw.shape[1] == 4:
                # assume ymin,xmin,ymax,xmax already in px
                boxes = boxes_raw / float(maxv)
            else:
                boxes = boxes_raw / float(maxv)

    # case B: [ymin,xmin,ymax,xmax,score,class] 포맷
    elif arr.shape[1] == 6:
        boxes = arr[:, :4]
        scores = arr[:, 4]
        classes = arr[:, 5].astype(np.int32)
    else:
        return np.array([]), np.array([]), np.array([])

    # 최종 필터링
    keep_mask = scores >= threshold
    if not np.any(keep_mask):
        return np.array([]), np.array([]), np.array([])

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    classes = classes[keep_mask]

    # 클래스별 NMS 및 결과 제한
    boxes, scores, classes = nms(boxes, scores, classes, IOU_THRESH)
    if boxes.size == 0:
        return np.array([]), np.array([]), np.array([])

    # --- 변경: 한 번에 가장 크게 보이는 객체 하나만 선택 ---
    # 면적(y 범위 * x 범위) 계산 (정규화된 값 사용)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = int(np.argmax(areas))
    box = boxes[largest_idx:largest_idx+1]
    score = scores[largest_idx:largest_idx+1]
    cls = classes[largest_idx:largest_idx+1]

    return box, score, cls
# ==========================================
# 3. 메인 실행 함수
# ==========================================
def main():
    # Interpreter가 준비되지 않았으면 친절히 종료
    if Interpreter is None:
        print("❌ tflite Interpreter / load_delegate를 찾을 수 없습니다. tflite-runtime 또는 tensorflow 설치 후 다시 실행하세요.")
        return

    # --- 모델 로드 (Edge TPU Delegate 포함) ---
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ 오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
        return

    try:
        delegate = load_delegate('libedgetpu.so.1')
        interpreter = Interpreter(model_path=str(model_path),
                              experimental_delegates=[delegate])
        print("✅ Edge TPU Delegate 로드 성공! 가속 모드로 실행합니다.")
    except Exception as e:
        print(f"⚠️ Edge TPU 로드 실패 ({e}), CPU 모드로 실행합니다.")
        interpreter = Interpreter(model_path=str(model_path))

    interpreter.allocate_tensors()

    # 입력/출력 텐서 정보 획득 - 한 번만 조회
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    output_idx = output_details[0]['index']

    # --- 입력 버퍼 한 번만 할당해 재사용 (메모리 할당 비용 절감) ---
    input_shape = input_details[0]['shape']  # e.g. [1, 320, 320, 3]
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0].get('quantization', (0.0, 0))
    input_scale, input_zero_point = input_quant if input_quant is not None else (0.0, 0)
    if input_scale == 0.0:
        input_scale = 1.0  # 안전장치

    # pre-allocated buffer
    input_buffer = np.zeros(input_shape, dtype=input_dtype)

    # 카메라 초기화 — 캡처 버퍼 최소화로 지연 줄이기
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 낮은 캡처 해상도 권장
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # 버퍼 사이즈를 1로 줄여 capture latency 완화 (파이카메라/드라이버 차이 있음)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        raise SystemExit("❌ 카메라를 열 수 없습니다. PiCam 연결 상태를 확인하세요.")

    print("\n--- 재활용품 분류 시스템 시작 (종료: 'q') ---")

    frame_count = 0
    last_boxes, last_scores, last_classes = np.array([]), np.array([]), np.array([])
    ema_latency = None

    # Local copy of dynamic skip value (avoid UnboundLocalError)
    process_every_n = PROCESS_EVERY_N
    # detection voting / persistence
    recent_votes = deque(maxlen=VOTE_WINDOW)
    persist_left = 0

    # 프레임 반복 루프
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 프레임은 항상 표시하지만 실제 추론은 process_every_n 기준으로 수행 -> latency 낮춤
        do_infer = (frame_count % process_every_n) == 0

        if do_infer:
            t0 = time.time()
            # --- 1. 전처리 (최소 복사) ---
            # 직접 리사이즈해 임시 배열 생성
            resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # 중요: RGB 변환

            # 양자화 모델이면 적절히 변환, 그렇지 않으면 float32 / 255
            if input_dtype == np.uint8 or input_dtype == np.int8:
                # float -> quantized: (img/255)/scale + zero_point
                tmp = (resized.astype(np.float32) / 255.0) / input_scale + input_zero_point
                # clip 및 dtype 변환, 그리고 preallocated 버퍼에 복사
                np.copyto(input_buffer[0], np.clip(tmp, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype))
            else:
                tmp = (resized.astype(np.float32) / 255.0).astype(np.float32)
                np.copyto(input_buffer[0], tmp)

            # --- 2. 추론 (Inference) ---
            interpreter.set_tensor(input_idx, input_buffer)
            infer_start = time.time()
            interpreter.invoke()
            infer_time = (time.time() - infer_start) * 1000.0

            # --- 3. 후처리 ---
            outputs = interpreter.get_tensor(output_idx)[0]
            print("DEBUG: output shape", outputs.shape, "min/max", outputs.min(), outputs.max())
            boxes, scores, classes = filter_boxes(outputs)

            last_boxes, last_scores, last_classes = boxes, scores, classes

            # vote/persistence update
            if classes.size:
                recent_votes.append(int(classes[0]))
                persist_left = PERSIST_FRAMES
            else:
                recent_votes.append(None)
                persist_left = max(0, persist_left - 1)

            # majority vote for stability
            vote_counts = Counter([v for v in recent_votes if v is not None])
            stable_cls = None
            if vote_counts:
                cls_candidate, cnt = vote_counts.most_common(1)[0]
                if cnt >= VOTE_REQUIRED:
                    stable_cls = cls_candidate

            # trigger all_2 when stable plastic detection and conditions met
            if ALL2_AVAILABLE and stable_cls is not None and stable_cls in (0, 1):
                now = time.time()
                if (now - trigger_last_time) > TRIGGER_COOLDOWN_SEC and not trigger_running:
                    label_name = LABEL_MAP.get(stable_cls, "")
                    material_code = 'A'   # map plastics -> 'A' (adjust in all_2 if needed)
                    is_dirty_flag = (label_name == 'Dirty_Plastic')

                    def _run_ctrl(material, dirty):
                        global trigger_running, trigger_last_time
                        trigger_running = True
                        try:
                            control_system_loop(material, dirty)
                        except Exception as ex:
                            print("[ERROR] control_system_loop failed:", ex)
                        trigger_last_time = time.time()
                        trigger_running = False

                    print(f"[TRIGGER] Detected stable {label_name} -> launching actuator (dirty={is_dirty_flag})")
                    Thread(target=_run_ctrl, args=(material_code, is_dirty_flag), daemon=True).start()

            # 지연 측정(프레임 전반)
            latency_ms = (time.time() - t0) * 1000.0
            if ema_latency is None:
                ema_latency = latency_ms
            else:
                ema_latency = 0.85 * ema_latency + 0.15 * latency_ms

            # 동적 프레임 스킵: 지연이 목표보다 크면 skip 늘리기 (최소/최대 제한)
            if ema_latency > TARGET_LATENCY_MS and process_every_n < MAX_PROCESS_N:
                process_every_n += 1
            elif ema_latency < TARGET_LATENCY_MS * 0.6 and process_every_n > MIN_PROCESS_N:
                process_every_n -= 1

            # 짧은 디버그 로그(원하면 줄일 수 있음)
            print(f"[INF] inference {infer_time:.1f}ms total {latency_ms:.1f}ms skip={process_every_n} detections={len(boxes)}")

        # --- 4. 시각화: 마지막으로 얻은 detection 사용 (매 프레임 그리기 가능) ---
        h, w, _ = frame.shape
        for box, score, cls in zip(last_boxes, last_scores, last_classes):
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)

            cls = int(cls)
            label_name = LABEL_MAP.get(cls, f"Class {cls}")
            label = f"{label_name}: {score:.2f}"
            color = COLOR_MAP.get(cls, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # FPS/Latency 출력
        loop_ms = (time.time() - loop_start) * 1000.0
        ema_text = f"{ema_latency:.0f}ms" if ema_latency is not None else "N/A"
        cv2.putText(frame, f"Loop: {loop_ms:.0f}ms  EMA: {ema_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not HEADLESS:
            cv2.imshow('Recycle Detection (Optimized)', frame)
        else:
            if int(time.time() * 10) % 30 == 0:
                cv2.imwrite('/tmp/recycle_last.jpg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_camera_threads(cap, interpreter, input_idx, output_idx, input_shape, input_dtype, input_scale, input_zero_point):
    frame_q = Queue(maxsize=1)   # 최신 프레임만 유지
    result_q = Queue(maxsize=1)

    stop_flag = {"stop": False}

    def capture_loop():
        while not stop_flag["stop"]:
            ret, frame = cap.read()
            if not ret:
                stop_flag["stop"] = True
                break
            # 'latest only' 정책: 큐가 가득하면 교체
            try:
                frame_q.put(frame, timeout=0.02)
            except:
                try:
                    _ = frame_q.get_nowait()
                    frame_q.put(frame)
                except:
                    pass

    def infer_loop():
        while not stop_flag["stop"]:
            try:
                frame = frame_q.get(timeout=0.05)
            except Empty:
                continue
            # 최소 전처리(복사 최소화)
            resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            if input_dtype in (np.uint8, np.int8):
                tmp = (resized.astype(np.float32) / 255.0) / input_scale + input_zero_point
                tmp = np.clip(tmp, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
            else:
                tmp = (resized.astype(np.float32) / 255.0).astype(np.float32)
            # set tensor & invoke
            interpreter.set_tensor(input_idx, np.expand_dims(tmp, axis=0))
            t0 = time.time()
            interpreter.invoke()
            infer_time = (time.time() - t0) * 1000.0
            outputs = interpreter.get_tensor(output_idx)[0]
            boxes, scores, classes = filter_boxes(outputs)
            # push latest result (drop old)
            try:
                result_q.put((boxes, scores, classes, infer_time), timeout=0.02)
            except:
                try:
                    _ = result_q.get_nowait()
                    result_q.put((boxes, scores, classes, infer_time))
                except:
                    pass

    t_cap = Thread(target=capture_loop, daemon=True)
    t_inf = Thread(target=infer_loop, daemon=True)
    t_cap.start()
    t_inf.start()
    return frame_q, result_q, stop_flag

if __name__ == "__main__":
    main()
