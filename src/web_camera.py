import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, Response

# --- 설정 ---
MODEL_PATH = "recycle.tflite"
LABELS = ['Can', 'Paper', 'Plastic']
CONF_THRESHOLD = 0.4  # 박스 그릴 기준 점수

app = Flask(__name__)

# --- 모델 로드 ---
print(f"모델 로딩 중: {MODEL_PATH}...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# --- 카메라 설정 ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. AI 추론 준비 (이미지 리사이즈)
            img = cv2.resize(frame, (input_shape[1], input_shape[2]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # 2. 추론 수행
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            # 3. 결과 그리기 (박스)
            for detection in output_data:
                confidence = detection[4]
                if confidence > CONF_THRESHOLD:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    class_conf = scores[class_id]
                    
                    if class_conf > CONF_THRESHOLD:
                        # 좌표 복원
                        x, y, w, h = detection[0:4]
                        x = int((x - w/2) * frame.shape[1])
                        y = int((y - h/2) * frame.shape[0])
                        w = int(w * frame.shape[1])
                        h = int(h * frame.shape[0])
                        
                        # 화면에 박스와 글자 그리기
                        label = f"{LABELS[class_id]}: {class_conf*100:.1f}%"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 4. 이미지를 웹 전송용(JPEG)으로 변환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 실시간 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0은 외부 접속을 허용한다는 뜻
    print("웹 서버 시작! 브라우저에서 접속하세요.")
    app.run(host='0.0.0.0', port=5000, debug=False)