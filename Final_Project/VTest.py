# VTest.py
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from AMC import ConstellationCNN, iq_to_constellation

# --- 1. 가상 신호 생성 (Digital Modulation 5종) ---
def add_awgn_noise(signal, noise_level):
    """
    사용자가 선택한 0~3단계 노이즈 레벨을 실제 AWGN 분산값으로 변환하여 추가.
    level 0: 0.00 (잡음 없음)
    level 1: 0.05 (약한 잡음)
    level 2: 0.15 (중간 잡음)
    level 3: 0.30 (강한 잡음)
    """
    noise_mapping = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.30}
    noise_power = noise_mapping.get(noise_level, 0.0)
    
    if noise_power > 0:
        signal[0, :] += np.random.normal(0, noise_power, signal.shape[1])
        signal[1, :] += np.random.normal(0, noise_power, signal.shape[1])
    return signal

def generate_bpsk(num_symbols=128, noise_level=0):
    phases = np.random.choice([0, np.pi], num_symbols)
    return add_awgn_noise(np.vstack((np.cos(phases), np.sin(phases))), noise_level), 'BPSK'

def generate_qpsk(num_symbols=128, noise_level=0):
    phases = np.random.choice([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4], num_symbols)
    return add_awgn_noise(np.vstack((np.cos(phases), np.sin(phases))), noise_level), 'QPSK'

def generate_8psk(num_symbols=128, noise_level=0):
    phases = np.random.choice([i * np.pi/4 for i in range(8)], num_symbols)
    return add_awgn_noise(np.vstack((np.cos(phases), np.sin(phases))), noise_level), '8PSK'

def generate_qam(num_symbols=128, m_ary=16, noise_level=0):
    """16QAM 및 64QAM 신호 생성"""
    sqrt_m = int(np.sqrt(m_ary))
    # 진폭 범위 설정 (예: 16QAM은 -3, -1, 1, 3)
    amplitudes = np.arange(-sqrt_m + 1, sqrt_m, 2)
    
    i_symbols = np.random.choice(amplitudes, num_symbols)
    q_symbols = np.random.choice(amplitudes, num_symbols)
    
    # 평균 전력을 1로 정규화 (다른 PSK 계열과 크기를 맞추기 위함)
    norm_factor = np.sqrt(np.mean(amplitudes**2) * 2)
    i_norm = i_symbols / norm_factor
    q_norm = q_symbols / norm_factor
    
    label = f'QAM{m_ary}'
    return add_awgn_noise(np.vstack((i_norm, q_norm)), noise_level), label

# --- 2. 시각화 및 모델 추론 ---
def plot_and_predict(model, classes, device, iq_signal, true_label, noise_level):
    img_matrix = iq_to_constellation(iq_signal)
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = classes[predicted_idx]
        confidence = probabilities[0][predicted_idx].item() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.scatter(iq_signal[0, :], iq_signal[1, :], alpha=0.6, color='blue')
    ax1.set_xlim([-2.5, 2.5]); ax1.set_ylim([-2.5, 2.5])
    ax1.axhline(0, color='black', linewidth=0.5); ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_title(f"Generated Signal (True: {true_label} / Noise Lvl: {noise_level})")
    ax1.grid(True)

    ax2.imshow(img_matrix, cmap='gray', origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
    ax2.set_title(f"CNN Input (Pred: {predicted_class} | {confidence:.1f}%)")
    
    color = 'green' if true_label == predicted_class else 'red'
    plt.suptitle(f"AMC Test Result: True [{true_label}] vs Pred [{predicted_class}]", color=color, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- 3. 메인 테스트 루프 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        with open('amc_classes.pkl', 'rb') as f:
            classes = pickle.load(f)
        model = ConstellationCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(torch.load('amc_model.pth', map_location=device, weights_only=True))
        print("✅ 성공: 학습된 모델 가중치를 불러왔습니다.")
    except Exception as e:
        print(f"❌ 오류: 모델 파일 로드 실패. ({e})")
        return

    # 함수 매핑 딕셔너리
    # 함수 매핑 딕셔너리 (파라미터 이름 일치시킴)
    generators = {
        '1': generate_bpsk,
        '2': generate_qpsk,
        '3': generate_8psk,
        # lambda의 첫 번째 인자 이름을 num_symbols로 명시적으로 맞춰줌
        '4': lambda num_symbols, noise_level: generate_qam(num_symbols=num_symbols, m_ary=16, noise_level=noise_level),
        '5': lambda num_symbols, noise_level: generate_qam(num_symbols=num_symbols, m_ary=64, noise_level=noise_level)
    }

    while True:
        print("\n" + "="*30)
        print("  [ AMC 가상 신호 검증 테스트 ]")
        print("="*30)
        print("1. BPSK")
        print("2. QPSK")
        print("3. 8PSK")
        print("4. QAM16")
        print("5. QAM64")
        print("6. 완전 랜덤 테스트 (신호 + 노이즈 자동 생성)")
        print("0. 종료")
        choice = input("선택 (0~6): ")
        
        if choice == '0':
            print("테스트를 종료합니다.")
            break
            
        if choice in generators:
            # 특정 신호 선택 시
            try:
                noise_input = int(input("노이즈 강도 선택 (0:없음, 1:약함, 2:보통, 3:강함): "))
                if noise_input not in [0, 1, 2, 3]:
                    print("잘못된 노이즈 단계입니다. 기본값 1(약함)로 진행합니다.")
                    noise_input = 1
            except ValueError:
                print("숫자를 입력해야 합니다. 기본값 1(약함)로 진행합니다.")
                noise_input = 1
                
            sig, true_lb = generators[choice](num_symbols=128, noise_level=noise_input)
            plot_and_predict(model, classes, device, sig, true_lb, noise_input)
            
        elif choice == '6':
            # 완전 랜덤 선택 시 (노이즈 입력 생략)
            random_key = random.choice(list(generators.keys()))
            random_noise = random.randint(0, 3) # 0~3 사이 랜덤 레벨
            
            print(f"\n🎲 랜덤 생성 중... (노이즈 레벨 {random_noise} 적용됨)")
            sig, true_lb = generators[random_key](num_symbols=128, noise_level=random_noise)
            plot_and_predict(model, classes, device, sig, true_lb, random_noise)
            
        else:
            print("잘못된 입력입니다. 다시 선택해 주세요.")

if __name__ == "__main__":
    main()
    