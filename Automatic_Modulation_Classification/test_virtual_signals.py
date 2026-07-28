import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from src.train_mobilenetv3 import (
    ConstellationCNN as MobileNetV3AMC,
    iq_to_constellation,
)
from src.train_resnet18 import ConstellationCNN as ResNet18AMC


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CLASSES_PATH = MODEL_DIR / "classes.pkl"
MODEL_CONFIGS = {
    "1": {
        "name": "MobileNetV3 Small",
        "model_class": MobileNetV3AMC,
        "weight_path": MODEL_DIR / "mobilenetv3_best.pth",
    },
    "2": {
        "name": "ResNet-18",
        "model_class": ResNet18AMC,
        "weight_path": MODEL_DIR / "resnet18_best.pth",
    },
}



# ============================================================
# 1. AWGN 노이즈 추가
# ============================================================
def add_awgn_noise(signal, noise_level):
    """
    0~3단계 노이즈 레벨 → 실제 AWGN 표준편차로 매핑
    level 0: σ=0.00 (잡음 없음)
    level 1: σ=0.05 (약한 잡음, ~26dB)
    level 2: σ=0.12 (중간 잡음, ~16dB)
    level 3: σ=0.25 (강한 잡음, ~10dB)
    """
    noise_mapping = {0: 0.0, 1: 0.05, 2: 0.12, 3: 0.25}
    noise_power   = noise_mapping.get(noise_level, 0.0)
    if noise_power > 0:
        signal[0, :] += np.random.normal(0, noise_power, signal.shape[1])
        signal[1, :] += np.random.normal(0, noise_power, signal.shape[1])
    return signal


# ============================================================
# 2. 가상 신호 생성 함수 (6종)
# ============================================================
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
    sqrt_m     = int(np.sqrt(m_ary))
    amplitudes = np.arange(-sqrt_m + 1, sqrt_m, 2)
    i_symbols  = np.random.choice(amplitudes, num_symbols)
    q_symbols  = np.random.choice(amplitudes, num_symbols)
    norm_factor = np.sqrt(np.mean(amplitudes**2) * 2)
    i_norm = i_symbols / norm_factor
    q_norm = q_symbols / norm_factor
    label  = f'QAM{m_ary}'
    return add_awgn_noise(np.vstack((i_norm, q_norm)), noise_level), label

def generate_pam4(num_symbols=128, noise_level=0):
    amplitudes = np.array([-3, -1, 1, 3])
    i_symbols  = np.random.choice(amplitudes, num_symbols)
    q_symbols  = np.zeros(num_symbols)
    norm_factor = np.sqrt(np.mean(amplitudes**2))
    i_norm = i_symbols / norm_factor
    return add_awgn_noise(np.vstack((i_norm, q_symbols)), noise_level), 'PAM4'

# 제너레이터 딕셔너리 (전역)
GENERATORS = {
    '1': generate_bpsk,
    '2': generate_qpsk,
    '3': generate_8psk,
    '4': generate_pam4,
    '5': lambda num_symbols, noise_level: generate_qam(num_symbols, m_ary=16,  noise_level=noise_level),
    '6': lambda num_symbols, noise_level: generate_qam(num_symbols, m_ary=64,  noise_level=noise_level),
}
LABEL_MAP = {'1':'BPSK','2':'QPSK','3':'8PSK','4':'PAM4','5':'QAM16','6':'QAM64'}


# ============================================================
# 3. 단일 신호 시각화 + 추론
# ============================================================
def plot_and_predict(model, classes, device, iq_signal, true_label, noise_level, model_name):
    img_matrix = iq_to_constellation(iq_signal)

    model.eval()
    with torch.no_grad():
        input_tensor  = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        output        = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_cls = classes[predicted_idx]
        confidence    = probabilities[predicted_idx].item() * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ── (a) IQ 산점도 ────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(iq_signal[0, :], iq_signal[1, :], alpha=0.6, color='royalblue', s=20)
    ax1.set_xlim([-2.5, 2.5]); ax1.set_ylim([-2.5, 2.5])
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_title(f'IQ Constellation\n(True: {true_label} / Noise Lvl: {noise_level})', fontsize=12)
    ax1.set_xlabel('In-Phase (I)'); ax1.set_ylabel('Quadrature (Q)')
    ax1.grid(True, alpha=0.4)

    # ── (b) CNN 입력 이미지 ──────────────────────────────
    ax2 = axes[1]
    ax2.imshow(img_matrix, cmap='inferno', origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
    ax2.set_title(f'CNN Input Image\n(Pred: {predicted_cls} | {confidence:.1f}%)', fontsize=12)
    ax2.set_xlabel('I'); ax2.set_ylabel('Q')

    # ── (c) 클래스별 확률 막대 ──────────────────────────
    ax3 = axes[2]
    probs  = probabilities.cpu().numpy() * 100
    colors = ['#2ecc71' if c == true_label else ('#e74c3c' if c == predicted_cls else '#95a5a6')
              for c in classes]
    bars = ax3.barh(classes, probs, color=colors, edgecolor='white', height=0.6)
    for bar, p in zip(bars, probs):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{p:.1f}%', va='center', fontsize=10)
    ax3.set_xlim([0, 115])
    ax3.set_xlabel('Probability (%)', fontsize=11)
    ax3.set_title('Class Probabilities\n(Green=True  Red=Pred)', fontsize=12)
    ax3.grid(True, axis='x', alpha=0.4)

    is_correct = (true_label == predicted_cls)
    color = 'green' if is_correct else 'red'
    verdict = ' CORRECT' if is_correct else ' WRONG'
    plt.suptitle(
        f'{model_name} | True [{true_label}] → Pred [{predicted_cls}] {verdict}',
        color=color, fontweight='bold', fontsize=14
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 메인 루프
# ============================================================
def select_model(device):
    print("\n사용할 모델을 선택한다.")
    print("  1. MobileNetV3 Small")
    print("  2. ResNet-18")
    choice = input("선택 (1~2): ").strip()

    config = MODEL_CONFIGS.get(choice)
    if config is None:
        raise ValueError("모델 선택은 1 또는 2여야 한다.")

    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)

    model = config["model_class"](num_classes=len(classes)).to(device)
    model.load_state_dict(
        torch.load(config["weight_path"], map_location=device, weights_only=True)
    )
    model.eval()
    return model, classes, config["name"]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model, classes, model_name = select_model(device)
        print(f"✅ {model_name} 가중치를 불러왔다.")
        print(f"   클래스: {classes}")
    except Exception as e:
        print(f"❌ 오류: 모델 파일 로드 실패. ({e})")
        return

    while True:
        print("\n" + "=" * 40)
        print("     [ AMC 가상 신호 검증 테스트 ]")
        print("=" * 40)
        print(" ─── 단일 신호 테스트 ───────────────────")
        print("  1. BPSK          2. QPSK")
        print("  3. 8PSK          4. PAM4")
        print("  5. QAM16         6. QAM64")
        print("  7. 완전 랜덤 (신호 + 노이즈 자동)")
        print(" ────────────────────────────────────────")
        print("  0. 종료")
        choice = input("선택 (0~7): ").strip()

        # ── 단일 신호 테스트 ────────────────────────────────
        if choice in GENERATORS:
            try:
                noise_input = int(input("노이즈 강도 (0:없음 / 1:약 / 2:중 / 3:강): "))
                if noise_input not in [0, 1, 2, 3]:
                    raise ValueError
            except ValueError:
                print("  잘못된 입력 → 기본값 1(약함) 적용")
                noise_input = 1
            sig, true_lb = GENERATORS[choice](num_symbols=128, noise_level=noise_input)
            plot_and_predict(model, classes, device, sig, true_lb, noise_input, model_name)

        elif choice == '7':
            rand_key   = random.choice(list(GENERATORS.keys()))
            rand_noise = random.randint(0, 3)
            print(f"\n🎲 [{LABEL_MAP[rand_key]}] 신호 / 노이즈 레벨 {rand_noise} 랜덤 선택")
            sig, true_lb = GENERATORS[rand_key](num_symbols=128, noise_level=rand_noise)
            plot_and_predict(model, classes, device, sig, true_lb, rand_noise, model_name)

        elif choice == '0':
            print("테스트를 종료합니다.")
            break

        else:
            print("  잘못된 입력입니다. 다시 선택해 주세요.")


if __name__ == "__main__":
    main()