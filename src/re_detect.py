from pathlib import Path
from queue import Queue, Empty
from threading import Thread
import cv2
import numpy as np
import os
import sys
import time
from collections import deque, Counter

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
CONF_THRESH = 0.35    # (변경) 더 높은 리콜을 위해 임계값을 낮춤 (0.35)
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


def filter_boxes(output_data, threshold=CONF_THRESH, max_results=50, output_details=None, input_size=INPUT_SIZE):
    """
    Parse model outputs robustly:
    - Dequantize when output_details provided
    - Accept two common layouts:
      * [cx, cy, w, h, obj_conf, class_prob0, ...]
      * [ymin, xmin, ymax, xmax, score, class_id]
    - Normalize coords/scores when needed and run class-wise NMS.
    """
    if output_data is None:
        return np.array([]), np.array([]), np.array([])

    arr = np.array(output_data)

    # dequantize if quantized output
    if output_details is not None:
        od_dtype = output_details.get('dtype', None)
        od_quant = output_details.get('quantization', (0.0, 0))
        od_scale, od_zp = od_quant if od_quant is not None else (0.0, 0)
        if od_dtype in (np.uint8, np.int8) and od_scale and od_scale != 0.0:
            arr = (arr.astype(np.float32) - float(od_zp)) * float(od_scale)
        else:
            arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)

    if arr.size == 0:
        return np.array([]), np.array([]), np.array([])

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    # Case A: cx,cy,w,h + obj_conf + class_probs  (>=7 columns)
    if arr.shape[1] >= 7:
        boxes_raw = arr[:, :4]
        obj_conf = arr[:, 4]
        class_probs = arr[:, 5:]
        # class_probs가 logits일 수 있으므로, 범위를 보고 softmax로 확률화
        if class_probs.size == 0:
            classes = np.zeros(obj_conf.shape[0], dtype=np.int32)
            class_conf = np.ones_like(obj_conf)
        else:
            # detect if class_probs look like probabilities (0..1) or logits/integers
            cp_min, cp_max = float(class_probs.min()), float(class_probs.max())
            if (cp_min < 0.0 - 1e-6) or (cp_max > 1.0 + 1e-3):
                # treat as logits -> stable softmax
                exps = np.exp(class_probs - np.max(class_probs, axis=1, keepdims=True))
                probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-12)
                class_id = np.argmax(probs, axis=1)
                class_conf = probs[np.arange(probs.shape[0]), class_id]
            else:
                class_id = np.argmax(class_probs, axis=1)
                class_conf = np.max(class_probs, axis=1)
            classes = class_id.astype(np.int32)

        # objectness might be logits or uncertain range -> sigmoid to prob if necessary
        oc_min, oc_max = float(obj_conf.min()), float(obj_conf.max())
        if (oc_min < 0.0 - 1e-6) or (oc_max > 1.0 + 1e-3):
            # apply sigmoid
            obj_prob = 1.0 / (1.0 + np.exp(-obj_conf.astype(np.float32)))
        else:
            obj_prob = obj_conf.astype(np.float32)

        scores = obj_prob * class_conf

        # heuristically normalize coords
        maxv = float(boxes_raw.max()) if boxes_raw.size else 1.0
        if maxv <= 1.0 + 1e-6:
            norm_boxes = boxes_raw.copy()
        elif maxv <= float(input_size) + 1e-3:
            norm_boxes = boxes_raw / float(input_size)
        elif maxv <= 255.0 + 1e-3:
            norm_boxes = boxes_raw / 255.0
        else:
            norm_boxes = boxes_raw / maxv

        # try cxcywh->yxyx or assume already yxyx; choose more-valid
        def cxcywh_to_yxyx(b):
            cx = b[:, 0]; cy = b[:, 1]; w = b[:, 2]; h = b[:, 3]
            y1 = cy - h / 2.0; x1 = cx - w / 2.0
            y2 = cy + h / 2.0; x2 = cx + w / 2.0
            return np.stack([y1, x1, y2, x2], axis=1)

        candA = cxcywh_to_yxyx(norm_boxes)
        candB = norm_boxes.copy()

        def valid_fraction(boxes):
            if boxes.size == 0:
                return 0.0
            y1 = boxes[:, 0]; x1 = boxes[:, 1]; y2 = boxes[:, 2]; x2 = boxes[:, 3]
            valid = (y1 >= 0.0) & (x1 >= 0.0) & (y2 > y1) & (x2 > x1) & (y2 <= 1.0) & (x2 <= 1.0)
            return float(valid.sum()) / boxes.shape[0]

        boxes = candA if valid_fraction(candA) >= valid_fraction(candB) else candB

    # Case B: ymin,xmin,ymax,xmax,score,class  (6 columns)
    elif arr.shape[1] == 6:
        boxes = arr[:, :4]
        scores = arr[:, 4]
        classes = arr[:, 5].astype(np.int32)
    else:
        return np.array([]), np.array([]), np.array([])

    # (주의) scores를 무작정 batch-max로 정규화하지 않음 — 위에서 sigmoid/softmax로 이미 정규화됨

    keep_mask = scores >= threshold
    if not np.any(keep_mask):
        return np.array([]), np.array([]), np.array([])

    boxes = boxes[keep_mask]; scores = scores[keep_mask]; classes = classes[keep_mask]

    boxes, scores, classes = nms(boxes, scores, classes, IOU_THRESH)
    if boxes.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Return up to max_results detections (sorted by score desc) - more robust than single largest only
    order = np.argsort(scores)[::-1]
    sel = order[:max_results]
    return boxes[sel], scores[sel], classes[sel]
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

    # --- 입력/출력 텐서 정보 획득 - 한 번만 조회
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    output_idx = output_details[0]['index']

    # 입력 관련 메타
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0].get('quantization', (0.0, 0))
    input_scale, input_zero_point = input_quant if input_quant is not None else (0.0, 0)
    if input_scale == 0.0:
        input_scale = 1.0

    # --- 디버그: 모델 입출력 정보 출력 (확인용) ---
    print("INPUT DETAILS:", input_details)
    print("OUTPUT DETAILS:", output_details)
    print("INPUT quant (scale, zp):", input_quant, "dtype:", input_dtype)
    print("LABEL_MAP:", LABEL_MAP)

    # --- 입력 버퍼 한 번만 할당해 재사용 (메모리 할당 비용 절감) ---
    input_shape = input_details[0]['shape']  # e.g. [1, 320, 320, 3]
    input_dtype = input_details[0]['dtype']
    input_quant = input_details[0].get('quantization', (0.0, 0))
    input_scale, input_zero_point = input_quant if input_quant is not None else (0.0, 0)
    if input_scale == 0.0:
        input_scale = 1.0
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
    # temporal smoothing / persistence settings
    VOTE_WINDOW = 5          # 최근 N개의 top-class를 투표
    VOTE_REQUIRED = 3        # 안정화 위해 필요한 다수(>=)
    PERSIST_FRAMES = 8       # detection 사라질 때 유지할 프레임 수
    recent_votes = deque(maxlen=VOTE_WINDOW)
    persist_left = 0

    # Local copy of dynamic skip value (avoid UnboundLocalError)
    process_every_n = PROCESS_EVERY_N

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
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # 중요: 모델이 RGB 기대하는 경우

            # --- 전처리: 양자화/입력 스케일에 따라 안전하게 처리 ---
            if input_dtype in (np.uint8, np.int8):
                # 모델이 0..1 기반인지(스케일 작음) 또는 0..255 기반인지(스케일 큼)에 따라 분기
                # 경험적으로 input_scale < 0.002 => 모델 expects 0..1 normalized input
                if input_scale is not None and input_scale > 0.002:
                    norm = resized.astype(np.float32)         # 0..255
                else:
                    norm = resized.astype(np.float32) / 255.0  # 0..1
                tmp = norm / input_scale + input_zero_point
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
            out_det = output_details[0]
            boxes, scores, classes = filter_boxes(outputs, output_details=out_det, input_size=INPUT_SIZE)
            # 디버그: 선택된 최종 객체 정보 출력
            if boxes.size:
                b0 = boxes[0]; cls0 = int(classes[0]); sc0 = float(scores[0])
                print(f"CHOICE: cls={cls0} name={LABEL_MAP.get(cls0,'?')} score={sc0:.3f} box={b0}")
                recent_votes.append(cls0)
                persist_left = PERSIST_FRAMES
            else:
                # push sentinel to vote history
                recent_votes.append(None)
                persist_left = max(0, persist_left - 1)

            # determine stable class using majority vote on recent_votes
            vote_counts = Counter([v for v in recent_votes if v is not None])
            stable_cls = None
            if vote_counts:
                cls_candidate, cnt = vote_counts.most_common(1)[0]
                if cnt >= VOTE_REQUIRED:
                    stable_cls = cls_candidate

            # pick boxes to display:
            if stable_cls is not None:
                # find best box for stable class if present
                idxs = np.where(classes == stable_cls)[0] if classes.size else np.array([], dtype=int)
                if idxs.size:
                    idx = int(np.argmax(scores[idxs]))
                    last_boxes = boxes[idx:idx+1]
                    last_scores = scores[idx:idx+1]
                    last_classes = classes[idx:idx+1]
                else:
                    # no current box but we're persisting -> keep previous
                    if persist_left > 0 and last_boxes.size:
                        pass
                    else:
                        last_boxes, last_scores, last_classes = np.array([]), np.array([]), np.array([])
            else:
                # no stable class: use latest detection (top result) if present, otherwise persist
                if boxes.size:
                    last_boxes, last_scores, last_classes = boxes[:1], scores[:1], classes[:1]
                else:
                    if persist_left > 0:
                        # keep previous last_* unchanged
                        pass
                    else:
                        last_boxes, last_scores, last_classes = np.array([]), np.array([]), np.array([])

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
    frame_q = Queue(maxsize=1)
    result_q = Queue(maxsize=1)
    stop_flag = {"stop": False}

    # pre-allocate input buffer once (avoid np.expand_dims on every frame)
    input_buffer = np.zeros(input_shape, dtype=input_dtype)
    out_det = interpreter.get_output_details()[0]

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
            resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            if input_dtype in (np.uint8, np.int8):
                if input_scale is not None and input_scale > 0.002:
                    norm = resized.astype(np.float32)
                else:
                    norm = resized.astype(np.float32) / 255.0
                tmp = norm / input_scale + input_zero_point
                tmp = np.clip(tmp, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
            else:
                tmp = (resized.astype(np.float32) / 255.0).astype(np.float32)

            # reuse preallocated input_buffer
            np.copyto(input_buffer[0], tmp)
            interpreter.set_tensor(input_idx, input_buffer)

            t0 = time.time()
            interpreter.invoke()
            infer_time = (time.time() - t0) * 1000.0
            outputs = interpreter.get_tensor(output_idx)[0]
            boxes, scores, classes = filter_boxes(outputs, output_details=out_det, input_size=INPUT_SIZE)
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

def run_model_io_test(model_path, test_img=None):
    """모델의 input/output 디테일과 한 번의 추론 결과 샘플을 출력합니다."""
    if Interpreter is None:
        print("Interpreter not available.")
        return
    try:
        delegate = None
        if load_delegate is not None:
            try:
                delegate = load_delegate('libedgetpu.so.1')
            except Exception:
                delegate = None
        interp = Interpreter(model_path=str(model_path), experimental_delegates=[delegate] if delegate else None)
        interp.allocate_tensors()
        idet = interp.get_input_details()[0]
        odet = interp.get_output_details()[0]
        print("INPUT:", idet)
        print("OUTPUT:", odet)
        if test_img:
            img = cv2.imread(test_img)
            img = cv2.resize(img, (idet['shape'][2], idet['shape'][1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if idet['dtype'] in (np.uint8, np.int8):
                scale, zp = idet.get('quantization', (0.0,0))
                if scale > 0.002:
                    inp = img.astype(np.float32)
                else:
                    inp = img.astype(np.float32) / 255.0
                inp = inp / (scale if scale!=0 else 1.0) + (zp if zp is not None else 0)
                inp = np.clip(inp, np.iinfo(idet['dtype']).min, np.iinfo(idet['dtype']).max).astype(idet['dtype'])
            else:
                inp = (img.astype(np.float32) / 255.0).astype(np.float32)
            inp = np.expand_dims(inp, axis=0)
            interp.set_tensor(idet['index'], inp)
            interp.invoke()
            out = interp.get_tensor(odet['index'])
            print("OUT shape", out.shape, "min/max", out.min(), out.max())
            print("OUT sample", out.flatten()[:64])
    except Exception as e:
        print("model io test failed:", e)

# re_detect에서 all_2의 제어 호출을 안전하게 사용
try:
    from all_2 import control_system_loop
    ALL2_AVAILABLE = True
except Exception as e:
    print("[WARN] all_2 module import failed — hardware trigger disabled.", e)
    ALL2_AVAILABLE = False

# 감지 -> 동작 트리거 설정
TRIGGER_COOLDOWN_SEC = 10.0   # 동작 후 재실행 방지 쿨다운
trigger_last_time = 0.0
trigger_running = False

if __name__ == "__main__":
    main()
