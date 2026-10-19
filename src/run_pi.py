import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# 설정
MODEL_PATH = "recycle.tflite"
LABELS = ['Can', 'Paper', 'Plastic']
CONF_THRESHOLD = 0.25  # 기준을 0.5 -> 0.25로 대폭 낮춤 (민감하게 반응)

print(f"모델 로딩 중: {MODEL_PATH}...")

# TFLite 인터프리터 로드
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# 카메라 켜기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("분류 시작! (최고 점수가 0.25를 넘으면 탐지됨)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 전처리
        img = cv2.resize(frame, (input_shape[1], input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 추론
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 디버깅: 현재 화면에서 가장 높은 점수 찾기
        max_conf = 0
        detected_label = ""
        
        for detection in output_data:
            confidence = detection[4] # 객체가 있을 확률
            if confidence > max_conf:
                scores = detection[5:]
                class_id = np.argmax(scores)
                max_conf = confidence * scores[class_id] # 최종 점수
                detected_label = LABELS[class_id]

        # 결과 출력 (점 대신 현재 최고 점수를 계속 보여줌)
        if max_conf > CONF_THRESHOLD:
            print(f"!!! 탐지 성공 !!! -> {detected_label} (정확도: {max_conf*100:.1f}%)")
        else:
            # 탐지 안 돼도 현재 최고 점수는 몇 점인지 보여줌 (답답함 해소)
            print(f"탐색 중... (현재 최고 점수: {max_conf*100:.1f}%)", end='\r')

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n종료합니다.")

cap.release()