# pca9685_smart_sorting_system_final_split.py

import time
import board
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# RPi.GPIO 라이브러리 추가 (초기화는 init_hardware에서 안전하게 수행)
try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    GPIO = None
    _HAS_GPIO = False

# --- 1. 설정값 및 채널 정의 ---
MIN_PULSE = 500
MAX_PULSE = 2500
INITIAL_ANGLE = 90

# Gate (채널 0, 4, 8) 및 이동량
GATE1_CH = 0
MOVE_GATE1_3 = 50
GATE2_CH = 4
MOVE_GATE2 = 30
GATE3_CH = 8

# Dual Servos (채널 15, 12) 및 안전 각도
SERVO_A_CH = 15
SERVO_B_CH = 12
MIN_SAFE_ANGLE = 5
MAX_SAFE_ANGLE = 175

# 속도 설정
SMOOTH_STEP = 2
SMOOTH_DELAY = 0.025  # Gate 움직임 속도
DUAL_STEP = 1
DUAL_DELAY = 0.025    # Dumper 움직임 속도

# RPi GPIO (워터 펌프)
WATER_PUMP_PIN = 27       # 예: BCM 27
RELAY_ON = 0              # Active Low (GPIO.LOW)
RELAY_OFF = 1             # GPIO.HIGH
WASH_DURATION = 3         # seconds

# --- 전역 변수 (지연 초기화) ---
servos = None
pca = None
_gpio_ready = False

def init_hardware():
    """서보/PCA9685와 RPi.GPIO 초기화 — 안전하게 여러 번 호출 가능."""
    global servos, pca, _gpio_ready
    if servos is not None:
        return True

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
    except Exception as e:
        print(f"[WARN] I2C/PCA9685 초기화 실패: {e}")
        pca = None

    if pca is None:
        print("[ERROR] PCA9685 초기화 실패 — 서보 기능 비활성화.")
        return False

    try:
        servos = {
            'gate1': servo.Servo(pca.channels[GATE1_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'gate2': servo.Servo(pca.channels[GATE2_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'gate3': servo.Servo(pca.channels[GATE3_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'servo_a': servo.Servo(pca.channels[SERVO_A_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
            'servo_b': servo.Servo(pca.channels[SERVO_B_CH], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE),
        }
        print(f"[INFO] 총 {len(servos)}개 서보 객체 생성 완료.")
    except Exception as e:
        print(f"[ERROR] 서보 객체 생성 실패: {e}")
        servos = None
        return False

    # 초기 각도 설정
    try:
        for s in servos.values():
            s.angle = INITIAL_ANGLE
        time.sleep(0.6)
    except Exception as e:
        print(f"[WARN] 서보 초기 각도 설정 실패: {e}")

    # RPi.GPIO 초기화
    if _HAS_GPIO:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(WATER_PUMP_PIN, GPIO.OUT, initial=GPIO.HIGH)
            _gpio_ready = True
            print(f"[INFO] RPi GPIO {WATER_PUMP_PIN} (Pump) 초기 설정 완료.")
        except Exception as e:
            _gpio_ready = False
            print(f"[WARN] RPi.GPIO 초기화 오류: {e}")
    else:
        _gpio_ready = False
        print("[WARN] RPi.GPIO 모듈을 찾을 수 없음 — 펌프 제어 비활성화.")

    return True

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
    if target_servo is None:
        print(f"[WARN] move_servo_smoothly: '{name}' 서보가 초기화되지 않음.")
        return
    step_val = step if end_angle > start_angle else -step
    start_int = int(start_angle)
    end_int = int(end_angle)
    # ensure endpoint is reached
    if step_val == 0:
        target_servo.angle = end_int
        return
    for angle in range(start_int, end_int + step_val, step_val):
        target_servo.angle = angle
        time.sleep(delay)
    target_servo.angle = end_int

def move_dual_servos_to_target(start_angle_a, end_angle_a, start_angle_b, end_angle_b, step, delay):
    """채널 15/12번 서보를 지정된 각도로 부드럽게 동시에 이동시킵니다."""
    if servos is None:
        print("[WARN] move_dual_servos_to_target: servos not initialized.")
        return
    servo_a = servos['servo_a']
    servo_b = servos['servo_b']

    step_a = step if end_angle_a > start_angle_a else -step
    step_b = step if end_angle_b > start_angle_b else -step

    print(f"[INFO] Dual Servo 이동: A({start_angle_a}°->{end_angle_a}°), B({start_angle_b}°->{end_angle_b}°) 시작.")
    path_a = list(range(int(start_angle_a), int(end_angle_a) + step_a, step_a))
    path_b = list(range(int(start_angle_b), int(end_angle_b) + step_b, step_b))
    max_len = max(len(path_a), len(path_b))
    for i in range(max_len):
        angle_a = path_a[i] if i < len(path_a) else end_angle_a
        angle_b = path_b[i] if i < len(path_b) else end_angle_b
        try:
            servo_a.angle = angle_a
            servo_b.angle = angle_b
        except Exception:
            pass
        time.sleep(delay)
    try:
        servo_a.angle = end_angle_a
        servo_b.angle = end_angle_b
    except Exception:
        pass

def run_dumper(command):
    """
    UP/DOWN 명령에 따라 Dual Servos를 이동시킵니다.
    """
    if servos is None:
        print("[WARN] run_dumper: servos not initialized.")
        return

    TARGET_A_UP = MAX_SAFE_ANGLE
    TARGET_B_DOWN = MIN_SAFE_ANGLE
    TARGET_A_DOWN = MIN_SAFE_ANGLE + 45
    TARGET_B_UP = MAX_SAFE_ANGLE - 45

    try:
        start_a = servos['servo_a'].angle
        start_b = servos['servo_b'].angle
    except Exception:
        start_a, start_b = INITIAL_ANGLE, INITIAL_ANGLE

    STEP, DELAY = DUAL_STEP, DUAL_DELAY

    if command.upper() == 'UP':
        print("[DUMPER] UP")
        move_dual_servos_to_target(start_a, TARGET_A_UP, start_b, TARGET_B_DOWN, STEP, DELAY)
    elif command.upper() == 'UP_C':
        print("[DUMPER] UP_C (center)")
        move_dual_servos_to_target(start_a, INITIAL_ANGLE, start_b, INITIAL_ANGLE, STEP, DELAY)
    elif command.upper() == 'DOWN':
        print("[DUMPER] DOWN")
        move_dual_servos_to_target(start_a, TARGET_A_DOWN, start_b, TARGET_B_UP, STEP, DELAY)
    elif command.upper() == 'DOWN_C':
        print("[DUMPER] DOWN_C (center)")
        move_dual_servos_to_target(start_a, INITIAL_ANGLE, start_b, INITIAL_ANGLE, STEP, DELAY)
    else:
        print(f"[WARN] run_dumper: Unknown command '{command}'")

def wash_action():
    """
    세척 기능: 워터 펌프를 지정 시간 동작시킵니다.
    """
    print(f"[WASH] Pump ON for {WASH_DURATION}s")
    if _gpio_ready and GPIO is not None:
        try:
            GPIO.output(WATER_PUMP_PIN, RELAY_ON)
            time.sleep(WASH_DURATION)
            GPIO.output(WATER_PUMP_PIN, RELAY_OFF)
            print("[WASH] Pump OFF")
        except Exception as e:
            print(f"[ERROR] wash_action GPIO failure: {e}")
            try:
                GPIO.output(WATER_PUMP_PIN, RELAY_OFF)
            except Exception:
                pass
    else:
        # fallback simulation/delay
        print("[WASH] GPIO not ready - simulating wash delay")
        time.sleep(WASH_DURATION)
    print("[WASH] 완료")

def run_gate_sorting(mode, material_choice, gate_list=None):
    """분류에 따라 Gate Servos를 움직입니다 ('OPEN' 또는 'CLOSE')."""
    if servos is None:
        print("[WARN] run_gate_sorting: servos not initialized.")
        return []

    target_g1_down = 90 - MOVE_GATE1_3
    target_g2_down = 90 - MOVE_GATE2
    target_g2_up = 90 + MOVE_GATE2
    target_g3_up = 90 + MOVE_GATE1_3
    start_angle = INITIAL_ANGLE

    if mode.upper() == 'OPEN':
        print(f"[GATE] OPEN: material={material_choice}")
        current_gate_names = []
        if material_choice == 'A':  # plastics
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
            print(f"[WARN] Unknown material code: '{material_choice}'")
            return []
        return current_gate_names

    elif mode.upper() == 'CLOSE' and gate_list:
        print("[GATE] CLOSE")
        for name in gate_list:
            s = servos.get(name)
            if s is None:
                continue
            try:
                current_angle = s.angle
            except Exception:
                current_angle = INITIAL_ANGLE
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE)
            time.sleep(0.1)
        time.sleep(0.5)
        return []
    return []

def reset_all_servos():
    """모든 서보를 90도로 부드럽게 초기화합니다. (Dumper 우선 복귀)"""
    if servos is None:
        print("[WARN] reset_all_servos: servos not initialized.")
        return
    print("[SYSTEM] Resetting all servos to 90°")
    dumper_names = ['servo_a', 'servo_b']
    gate_names = ['gate1', 'gate2', 'gate3']

    for name in dumper_names:
        s = servos[name]
        try:
            current_angle = int(s.angle)
        except Exception:
            current_angle = INITIAL_ANGLE
        if current_angle != INITIAL_ANGLE:
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE, step=DUAL_STEP, delay=DUAL_DELAY)
            time.sleep(0.1)

    for name in gate_names:
        s = servos[name]
        try:
            current_angle = int(s.angle)
        except Exception:
            current_angle = INITIAL_ANGLE
        if current_angle != INITIAL_ANGLE:
            move_servo_smoothly(s, name, current_angle, INITIAL_ANGLE)
            time.sleep(0.1)
    print("[SYSTEM] Reset complete.")

# ----------------------------------------------------
# 5. 통합 제어 루프
# ----------------------------------------------------
def control_system_loop(material, is_dirty):
    """재활용 물질(material 코드)과 오염 여부에 따라 시퀀스를 실행."""
    ok = init_hardware()
    if not ok:
        print("[ERROR] hardware init failed - aborting control loop.")
        return

    print(f"\n=== 제어 시퀀스 시작: material={material} dirty={is_dirty} ===")
    reset_all_servos()
    time.sleep(0.4)
    open_gates = []

    try:
        if is_dirty:
            print("[SEQ] Dirty item: move to wash station")
            run_dumper('UP')
            time.sleep(1.0)
            wash_action()
            print("[SEQ] Returning to sorting level")
            run_dumper('UP_C')
            time.sleep(0.5)
            open_gates = run_gate_sorting('OPEN', material)
            run_dumper('DOWN')
            run_dumper('DOWN_C')
            run_gate_sorting('CLOSE', material, gate_list=open_gates)
        else:
            print("[SEQ] Clean item: direct sorting")
            open_gates = run_gate_sorting('OPEN', material)
            run_dumper('DOWN')
            run_dumper('DOWN_C')
            run_gate_sorting('CLOSE', material, gate_list=open_gates)
    except Exception as e:
        print(f"[ERROR] control sequence failed: {e} — performing safe reset.")
        reset_all_servos()

    print("=== 제어 시퀀스 종료 ===")

# ----------------------------------------------------
# 6. 메인 (테스트용)
# ----------------------------------------------------
if __name__ == '__main__':
    ok = init_hardware()
    if not ok:
        print("[FATAL] hardware init failed — exiting.")
        raise SystemExit(1)

    test_scenarios = [
        ('B', True),
        ('C', False),
        ('A', False),
        ('D', False),
    ]

    try:
        while True:
            for material_code, is_dirty in test_scenarios:
                control_system_loop(material_code, is_dirty)
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n[EXIT] KeyboardInterrupt - final reset.")
        reset_all_servos()
    finally:
        if _HAS_GPIO and GPIO is not None:
            try:
                GPIO.cleanup()
                print("[CLEANUP] RPi GPIO cleaned up.")
            except Exception:
                pass
        print("[EXIT] Program terminated.")