# pca9685_smart_sorting_system_final_split.py

import time
import board
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# --- 1. 설정값 및 채널 정의 ---
MIN_PULSE = 500
MAX_PULSE = 2500
INITIAL_ANGLE = 90

# Gate (채널 0, 4, 8) 및 이동량
GATE1_CH = 0; MOVE_GATE1_3 = 50
GATE2_CH = 4; MOVE_GATE2 = 30
GATE3_CH = 8

# Dual Servos (채널 15, 12) 및 안전 각도
SERVO_A_CH = 15
SERVO_B_CH = 12
MIN_SAFE_ANGLE = 5
MAX_SAFE_ANGLE = 175

# 속도 설정
SMOOTH_STEP = 2
SMOOTH_DELAY = 0.025 # Gate 움직임 속도
DUAL_STEP = 1
DUAL_DELAY = 0.025    # Dumper 움직임 속도

# --- 전역 servos 변수 (지연 초기화) ---
servos = None
pca = None

def init_hardware():
    """서보/PCA9685 초기화 — import 시 자동으로 실행되지 않도록 분리"""
    global servos, pca
    if servos is not None:
        return
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        servos = {
            'gate1': servo.Servo(pca.channels[GATE1_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'gate2': servo.Servo(pca.channels[GATE2_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'gate3': servo.Servo(pca.channels[GATE3_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'servo_a': servo.Servo(pca.channels[SERVO_A_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'servo_b': servo.Servo(pca.channels[SERVO_B_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
        }
        for s in servos.values():
            s.angle = INITIAL_ANGLE
        time.sleep(0.6)
        print("PCA9685 및 servos 초기화 완료.")
    except Exception as e:
        servos = None
        pca = None
        print(f"[WARN] 하드웨어 초기화 실패: {e}")

# 1. I2C 및 PCA9685 초기 설정
try:
    i2c = busio.I2C(board.SCL, board.SDA) 
    pca = PCA9685(i2c)
    pca.frequency = 50
    print("PCA9685 초기 설정 완료.")
except Exception as e:
    print(f"I2C 또는 PCA9685 초기화 오류: {e}. 코드를 실행할 수 없습니다.")
    exit()

# 2. 서보 모터 객체 생성 (총 5개 서보 통합)
servos = {
    'gate1': servo.Servo(pca.channels[GATE1_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
    'gate2': servo.Servo(pca.channels[GATE2_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
    'gate3': servo.Servo(pca.channels[GATE3_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
    'servo_a': servo.Servo(pca.channels[SERVO_A_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
    'servo_b': servo.Servo(pca.channels[SERVO_B_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
}
print(f"총 {len(servos)}개 서보 객체 생성 완료.")

# --- 초기 각도 설정 ---
for s in servos.values():
    s.angle = INITIAL_ANGLE
time.sleep(1) 

# --- 유틸리티 및 제어 함수 ---

def get_channel_num(name):
    if name == 'gate1': return GATE1_CH
    if name == 'gate2': return GATE2_CH
    if name == 'gate3': return GATE3_CH
    if name == 'servo_a': return SERVO_A_CH
    if name == 'servo_b': return SERVO_B_CH
    return -1

def move_servo_smoothly(target_servo, name, start_angle, end_angle, step=SMOOTH_STEP, delay=SMOOTH_DELAY):
    """단일 서보를 부드럽게 이동시키는 범용 함수."""
    step = step if end_angle > start_angle else -step
    start_int = int(start_angle)
    end_int = int(end_angle)

    for angle in range(start_int, end_int + step, step):
        target_servo.angle = angle
        time.sleep(delay)
    
    target_servo.angle = end_int

# --- 유틸리티 및 제어 함수 (수정) ---
# NOTE: 이 함수의 인수는 6개입니다.
def move_dual_servos_to_target(start_angle_a, end_angle_a, start_angle_b, end_angle_b, step, delay):
    """채널 15/12번 서보를 지정된 각도로 부드럽게 동시에 이동시킵니다."""
    servo_a, servo_b = servos['servo_a'], servos['servo_b']

    # ... (함수 내부 로직은 이전과 동일하게 6개 인수를 사용) ...
    
    # [Rest of move_dual_servos_to_target body]
    step_a = step if end_angle_a > start_angle_a else -step
    step_b = step if end_angle_b > start_angle_b else -step

    print(f"  -> Dual Servo 이동: A({start_angle_a}°->{end_angle_a}°), B({start_angle_b}°->{end_angle_b}°) 이동 시작.")

    path_a = list(range(int(start_angle_a), int(end_angle_a) + step_a, step_a))
    path_b = list(range(int(start_angle_b), int(end_angle_b) + step_b, step_b))
    max_len = max(len(path_a), len(path_b))
    
    for i in range(max_len):
        angle_a = path_a[i] if i < len(path_a) else end_angle_a
        angle_b = path_b[i] if i < len(path_b) else end_angle_b
        
        servo_a.angle = angle_a
        servo_b.angle = angle_b
        time.sleep(delay)
        
    servo_a.angle = end_angle_a
    servo_b.angle = end_angle_b
    # ...

def run_dumper(command):
    """
    UP/DOWN 명령에 따라 Dual Servos를 이동시킵니다.
    """
    
    TARGET_A_UP = MAX_SAFE_ANGLE
    TARGET_B_DOWN = MIN_SAFE_ANGLE
    TARGET_A_DOWN = MIN_SAFE_ANGLE
    TARGET_B_UP = MAX_SAFE_ANGLE
    
    # 현재 각도를 읽어와 시작점으로 사용
    try:
        start_a = servos['servo_a'].angle
        start_b = servos['servo_b'].angle
    except:
        start_a, start_b = INITIAL_ANGLE, INITIAL_ANGLE
        
    STEP, DELAY = DUAL_STEP, DUAL_DELAY
    
    if command.upper() == 'UP': 
        print("\n[DUMPER: UP] A: 90->175, B: 90->5 (단일 이동)")
        # 6개 인수를 정확히 전달
        move_dual_servos_to_target(start_a, TARGET_A_UP, start_b, TARGET_B_DOWN, STEP, DELAY)
        
    elif command.upper() == 'UP_C': 
        print("\n[DUMPER: UP_C] Center 복귀 (175->90 / 5->90)")
        # 6개 인수를 정확히 전달
        move_dual_servos_to_target(start_a, INITIAL_ANGLE, start_b, INITIAL_ANGLE, STEP, DELAY)
        
    # ... (DOWN, DOWN_C 로직도 동일하게 6개 인수를 전달)
    elif command.upper() == 'DOWN': 
        print("\n[DUMPER: DOWN] A: 90->5, B: 90->175 (단일 이동)")
        move_dual_servos_to_target(start_a, TARGET_A_DOWN, start_b, TARGET_B_UP, STEP, DELAY)
        
    elif command.upper() == 'DOWN_C': 
        print("\n[DUMPER: DOWN_C] Center 복귀 (5->90 / 175->90)")
        move_dual_servos_to_target(start_a, INITIAL_ANGLE, start_b, INITIAL_ANGLE, STEP, DELAY)

def run_gate_sorting(mode, material_choice, gate_list=None):
    """분류에 따라 Gate Servos를 움직입니다 ('OPEN' 또는 'CLOSE')."""
    
    target_g1_down = 90 - MOVE_GATE1_3
    target_g2_down = 90 - MOVE_GATE2    
    target_g2_up = 90 + MOVE_GATE2      
    target_g3_up = 90 + MOVE_GATE1_3    

    start_angle = INITIAL_ANGLE
    
    if mode.upper() == 'OPEN':
        print(f"--- Gate OPEN: {material_choice} ---")
        current_gate_names = []
        
        # 1. GATE OPEN (분류) 로직
        if material_choice == 'A': 
            move_servo_smoothly(servos['gate1'], 'gate1', start_angle, target_g1_down) 
            move_servo_smoothly(servos['gate2'], 'gate2', start_angle, target_g2_down)
            current_gate_names.extend(['gate1', 'gate2'])
        elif material_choice == 'B': 
            move_servo_smoothly(servos['gate2'], 'gate2', start_angle, target_g2_down)
            current_gate_names.append('gate2')
        elif material_choice == 'C': 
            move_servo_smoothly(servos['gate2'], 'gate2', start_angle, target_g2_up)
            current_gate_names.append('gate2')
        elif material_choice == 'D': 
            move_servo_smoothly(servos['gate2'], 'gate2', start_angle, target_g2_up)
            move_servo_smoothly(servos['gate3'], 'gate3', start_angle, target_g3_up)
            current_gate_names.extend(['gate2', 'gate3'])
        else:
            print(f"경고: 알 수 없는 물질 코드 '{material_choice}'입니다.")
            return []
            
        return current_gate_names 

    elif mode.upper() == 'CLOSE' and gate_list:
        print("--- Gate CLOSE ---")
        # 2. 90도로 복귀 (Gate 닫기)
        for name in gate_list:
            s = servos[name]
            try:
                current_angle = s.angle
            except:
                current_angle = INITIAL_ANGLE
                
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE)
            time.sleep(0.1) 
        
        time.sleep(0.5) 
        return []
    
    return []

def wash_action():
    """
    세척 기능을 시뮬레이션하는 함수입니다.
    """
    print("--- 2. WASHING... (3 seconds) ---")
    
    # 실제 세척 동작 코드가 들어갈 위치
    pass 
    
    time.sleep(3) # 세척이 완료될 때까지 3초 대기
    print("--- WASHING 완료 ---")


def reset_all_servos():
    """모든 서보를 90도로 부드럽게 초기화합니다. (Dumper 우선 복귀)"""
    print("\n[SYSTEM RESET] 모든 서보 90° 복귀 중...")
    
    dumper_names = ['servo_a', 'servo_b']
    gate_names = ['gate1', 'gate2', 'gate3']
    
    # 1. DUMPER SERVOS 먼저 복귀 (느린 속도)
    for name in dumper_names:
        s = servos[name]
        try:
            current_angle = int(s.angle)
        except:
            current_angle = INITIAL_ANGLE
            
        if current_angle != INITIAL_ANGLE:
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE, step=DUAL_STEP, delay=DUAL_DELAY)
            time.sleep(0.1)
            
    # 2. GATE SERVOS 복귀 (빠른 속도)
    for name in gate_names:
        s = servos[name]
        try:
            current_angle = int(s.angle)
        except:
            current_angle = INITIAL_ANGLE
            
        if current_angle != INITIAL_ANGLE:
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE)
            time.sleep(0.1) 
            
    print("[SYSTEM RESET] 완료.")

# ----------------------------------------------------
# 5. 새로운 통합 제어 로직
# ----------------------------------------------------
def control_system_loop(material, is_dirty):
    """재활용 물질과 오염 여부에 따라 제어 시퀀스를 실행합니다."""
    global servos
    # 하드웨어 초기화 지연 수행
    if servos is None:
        init_hardware()
        if servos is None:
            print("[ERROR] 서보 초기화 실패 — 하드웨어 연결/권한 확인 필요.")
            return
    
    print(f"\n==================================================")
    print(f"분류 시작: {material} (Dirty: {is_dirty})")
    
    # 1. 시스템 초기화
    reset_all_servos()
    time.sleep(0.5)
    open_gates = [] # 열린 게이트 목록 저장

    try:
        if is_dirty:
            print(">>> [DIRTY] 오염 물질 감지: 세척 후 분류 시퀀스 <<<")
            
            # 1. UP: 세척 스테이션으로 이동
            run_dumper('UP')
            time.sleep(1) 
            
            # 2. WASH: 세척 동작
            wash_action() # 3초 대기
            
            # 3. 90° 복귀: 듀얼 서보가 중앙 분류 레벨로 복귀 
            print("--- 3. Dual Servo 90° 복귀 (Sorting Level로 이동) ---")
            run_dumper('UP_C') 
            time.sleep(0.5)
            
            # 4. SORT & DOWN (배출) - 사용자 요청 최종 로직
            print("--- 4. SORTING & DOWN (Active Discharge) ---")
            
            # 4a. Gate Open
            open_gates = run_gate_sorting('OPEN', material)

            # 4b. Dumper DOWN (밀어내기 1차 이동: 90->5)
            run_dumper('DOWN')
            
            # 4c. Dumper CENTER RETURN (밀어내기 복귀: 5->90)
            run_dumper('DOWN_C')
            
            # 4d. Gate Close
            run_gate_sorting('CLOSE', material, gate_list=open_gates)
            
        else: # Not Dirty (Clean)
            print(">>> [CLEAN] 청결 물질 감지: 즉시 분류 시퀀스 <<<")
            
            # 1. SORT & DOWN (배출)
            print("--- 1. SORTING & DOWN (Active Discharge) ---")

            # 1a. Gate Open
            open_gates = run_gate_sorting('OPEN', material)

            # 1b. Dumper DOWN (밀어내기 1차 이동: 90->5)
            run_dumper('DOWN')
            
            # 1c. Dumper CENTER RETURN (밀어내기 복귀: 5->90)
            run_dumper('DOWN_C')

            # 1d. Gate Close
            run_gate_sorting('CLOSE', material, gate_list=open_gates)
            
    except Exception as e:
        print(f"\n🔴 오류 발생: {e}. 안전 복귀를 시작합니다.")
        reset_all_servos() 

    print("==================================================")


# 6. 메인 실행
if __name__ == '__main__':
    init_hardware()
    # 테스트 시나리오 정의: (물질 코드, 오염 여부)
    test_scenarios = [
        ('B', True),   
        ('C', False),  
        ('A', True),   
        ('D', False)   
    ]
    
    try:
        while True:
            for material_code, is_dirty in test_scenarios:
                control_system_loop(material_code, is_dirty)
                time.sleep(2) 
    except KeyboardInterrupt:
        print("\n프로그램 종료. 최종 리셋을 수행합니다.")
        reset_all_servos()
        print("프로그램이 안전하게 종료되었습니다.")