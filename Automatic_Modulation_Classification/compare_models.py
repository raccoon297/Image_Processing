import json
import torch
from pathlib import Path
import time
import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from thop import profile

# 두 파일에서 모델 클래스를 각각 임포트 (파일이 같은 폴더에 있어야 함)
from src.train_mobilenetv3 import ConstellationCNN as MobileNet_AMC
from src.train_resnet18 import ConstellationCNN as ResNet_AMC


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
MOBILE_METRICS_PATH = RESULT_DIR / "mobilenetv3_metrics.json"
RESNET_METRICS_PATH = RESULT_DIR / "resnet18_metrics.json"

# ==========================================
# 0. 한글 폰트 설정 (OS 자동 인식)
# ==========================================
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 성능 측정 유틸리티 함수
# ==========================================


def load_accuracy(metrics_path):
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"학습 지표 파일을 찾을 수 없다: {metrics_path}\n"
            "먼저 해당 모델의 학습 스크립트를 실행해야 한다."
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics["best_val_accuracy"]), metrics


def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def measure_inference_time(model, device, input_shape=(1, 1, 32, 32), iterations=100):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
            
    if device.type == 'cuda': torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations): _ = model(dummy_input)
            
    if device.type == 'cuda': torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    return ((end_time - start_time) / iterations) * 1000

# ==========================================
# 2. 메인 로직: 성능 측정 및 터미널 출력
# ==========================================
def main():
    RESULT_DIR.mkdir(exist_ok=True)

    # 엣지 디바이스(드론 환경) 시뮬레이션을 위한 CPU 강제 할당
    device = torch.device('cpu')
    ## device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 측정 환경: {device}")
    
    mobilenet = MobileNet_AMC(num_classes=6)
    resnet = ResNet_AMC(num_classes=6)
    dummy_input = torch.randn(1, 1, 32, 32)
    
    print("⏳ 연산량, 파라미터 및 추론 속도 계산 중...")
    macs_m, params_m = profile(mobilenet, inputs=(dummy_input,), verbose=False)
    macs_r, params_r = profile(resnet, inputs=(dummy_input,), verbose=False)
    
    # MACs를 GFLOPs 단위로 변환 (10^9)
    flops_m = (macs_m * 2) / 1e9
    flops_r = (macs_r * 2) / 1e9
    
    size_m = get_model_size_mb(mobilenet)
    size_r = get_model_size_mb(resnet)
    
    time_m = measure_inference_time(mobilenet, device)
    time_r = measure_inference_time(resnet, device)
    
    acc_m, mobile_metrics = load_accuracy(MOBILE_METRICS_PATH)
    acc_r, resnet_metrics = load_accuracy(RESNET_METRICS_PATH)

    # ── 터미널 테이블 출력 (요청하신 포맷 적용) ──
    print("\n" + "="*66)
    print(f"비교 지표{' '*13} | MobileNetV3 (AMC1) | ResNet-18 (AMC2)")
    print("-" * 66)
    print(f"파라미터 수 (Params){' '*3} | {params_m/1e6:>10.2f} M       | {params_r/1e6:>10.2f} M")
    print(f"연산량 (GFLOPs){' '*8} | {flops_m:>10.4f} G       | {flops_r:>10.4f} G")
    print(f"메모리 용량 (MB){' '*7} | {size_m:>10.2f} MB      | {size_r:>10.2f} MB")
    print(f"평균 추론 시간 (ms){' '*4} | {time_m:>10.2f} ms      | {time_r:>10.2f} ms")
    print(f"10dB 이상 정확도 (%){' '*3} | {acc_m:>10.2f} %       | {acc_r:>10.2f} %")
    print("="*66 + "\n")

    # ==========================================
    # 3. 바 차트 시각화 (model_comparison.png)
    # ==========================================
    labels = ['Params (M)', 'FLOPs (G)', 'Size (MB)', 'Inf. Time (ms)', 'Accuracy (%)']
    mobile_stats = [params_m/1e6, flops_m, size_m, time_m, acc_m]
    resnet_stats = [params_r/1e6, flops_r, size_r, time_r, acc_r]

    x = np.arange(len(labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    rects1 = ax1.bar(x - width/2, mobile_stats, width, label='MobileNetV3', color='#4CAF50')
    rects2 = ax1.bar(x + width/2, resnet_stats, width, label='ResNet-18', color='#F44336')

    ax1.set_yscale('log')
    ax1.set_ylabel('Value (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('MobileNetV3 vs ResNet-18 Performance Trade-off', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax1.legend(fontsize=12)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            text = f'{height:.1f}' if height >= 1 else f'{height:.3f}'
            ax1.annotate(text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig1.tight_layout()
    fig1.savefig(RESULT_DIR / 'model_comparison.png')
    print("✅ 'model_comparison.png' (로그 스케일 바 차트) 저장 완료!")

    # ==========================================
    # 4. 와이드 대시보드 시각화 (model_dashboard_auto.png)
    # ==========================================
    metrics = {
        'labels': ['파라미터 수', '연산량\n(FLOPs)', '모델 메모리', '추론 시간', 'Best val\nAcc'],
        'mobile': [round(params_m/1e6, 2), round(flops_m, 4), round(size_m, 2), round(time_m, 2), round(acc_m, 2)],
        'resnet': [round(params_r/1e6, 2), round(flops_r, 4), round(size_r, 2), round(time_r, 2), round(acc_r, 2)],
        'units': ['M', 'G', 'MB', 'ms', '%']
    }

    ratio_texts, ratio_colors = [], []
    for i in range(4):
        if i != 3: 
            times = metrics['resnet'][i] / metrics['mobile'][i] if metrics['mobile'][i] > 0 else 1
            ratio_texts.append(f"{times:.1f}x 절감")
            ratio_colors.append('#3730A3')
        else: 
            if metrics['mobile'][i] <= metrics['resnet'][i]:
                times = metrics['resnet'][i] / metrics['mobile'][i] if metrics['mobile'][i] > 0 else 1
                ratio_texts.append(f"{times:.1f}x 빠름")
                ratio_colors.append('#064E3B') 
            else:
                times = metrics['mobile'][i] / metrics['resnet'][i] if metrics['resnet'][i] > 0 else 1
                ratio_texts.append(f"{times:.1f}x 느림")
                ratio_colors.append('#991B1B') 

    acc_diff = metrics['mobile'][4] - metrics['resnet'][4]
    ratio_texts.append(f"{acc_diff:.1f}%p")
    ratio_colors.append('#B45309') 

    def calc_score(val_m, val_r):
        min_val = min(val_m, val_r)
        return (min_val / val_m), (min_val / val_r)

    score_p_m, score_p_r = calc_score(metrics['mobile'][0], metrics['resnet'][0])
    score_f_m, score_f_r = calc_score(metrics['mobile'][1], metrics['resnet'][1])
    score_mem_m, score_mem_r = calc_score(metrics['mobile'][2], metrics['resnet'][2])
    score_t_m, score_t_r = calc_score(metrics['mobile'][3], metrics['resnet'][3])

    mobile_stability = 1.0 / (1.0 + float(mobile_metrics['best_val_loss']))
    resnet_stability = 1.0 / (1.0 + float(resnet_metrics['best_val_loss']))
    mobile_scores = [acc_m/100, score_p_m, score_f_m, score_mem_m, score_t_m, mobile_stability]
    resnet_scores = [acc_r/100, score_p_r, score_f_r, score_mem_r, score_t_r, resnet_stability]
    mobile_scores += mobile_scores[:1]
    resnet_scores += resnet_scores[:1]

    BG_MAIN, BG_PANEL = '#FAF8F5', '#FFFFFF'
    TEXT_MAIN, TEXT_SUB = '#2D3748', '#718096'
    COLOR_MOBILE, COLOR_RESNET = '#4F46E5', '#E11D48'

    fig2 = plt.figure(figsize=(15, 7), facecolor=BG_MAIN, dpi=150)

    ax_radar = fig2.add_axes([0.05, 0.1, 0.4, 0.8], polar=True, facecolor=BG_PANEL)
    radar_labels = ['정확도', '파라미터 효율', '연산 효율', '메모리 효율', '추론 속도', '수렴 안정성']
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(0)
    ax_radar.spines['polar'].set_visible(False)
    ax_radar.grid(False) 

    for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax_radar.plot(angles, [level]*len(angles), color='#E2E8F0', linewidth=1.5, linestyle='solid')
    for angle in angles[:-1]:
        ax_radar.plot([angle, angle], [0, 1], color='#E2E8F0', linewidth=1.5, linestyle='solid')

    ax_radar.plot(angles, mobile_scores, color=COLOR_MOBILE, linewidth=2.5, linestyle='solid', label='MobileNetV3')
    ax_radar.fill(angles, mobile_scores, color=COLOR_MOBILE, alpha=0.15)
    ax_radar.plot(angles, resnet_scores, color=COLOR_RESNET, linewidth=2.5, linestyle='--', label='ResNet-18')

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, color=TEXT_SUB, fontsize=12, fontweight='bold')
    ax_radar.set_yticklabels([]) 
    
    # 종합 역량 평가
    ax_radar.text(0.5, 1.15, "Comprehensive Capability Evaluation", transform=ax_radar.transAxes, ha='center', color=TEXT_MAIN, fontsize=15, fontweight='bold')
    ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, labelcolor=TEXT_MAIN, fontsize=12)
    ax_radar.text(0.5, -0.25, "* 효율 지표는 상대 정규화, 수렴 안정성은 1/(1+Best Val Loss)", transform=ax_radar.transAxes, ha='center', color=TEXT_SUB, fontsize=10)

    ax_table = fig2.add_axes([0.52, 0.1, 0.43, 0.8], facecolor=BG_PANEL)
    ax_table.axis('off') 
    
    # 핵심 지표 비교표
    ax_table.text(0.05, 0.96, "Core Performance Metrics", color=TEXT_MAIN, fontsize=15, fontweight='bold', transform=ax_table.transAxes)
    ax_table.plot([0.05, 0.95], [0.92, 0.92], color='#CBD5E1', linewidth=1.5, transform=ax_table.transAxes)

    cols_x = [0.05, 0.35, 0.60, 0.84]
    for x, h in zip(cols_x, ['지 표', 'MobileNetV3', 'ResNet-18', '비 율']):
        ax_table.text(x, 0.85, h, color=TEXT_SUB, fontsize=12, fontweight='bold', transform=ax_table.transAxes)

    ax_table.plot([0.05, 0.95], [0.80, 0.80], color='#E2E8F0', linewidth=1.5, transform=ax_table.transAxes)

    y_positions = [0.68, 0.53, 0.38, 0.23, 0.08]
    colors_mobile, colors_resnet, text_colors = ['#4338CA']*5, ['#BE185D']*5, ['#FFFFFF']*5

    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            ax_table.add_patch(patches.Rectangle((0.02, y-0.06), 0.96, 0.12, transform=ax_table.transAxes, color='#F8F9FA', zorder=0))

        ax_table.text(cols_x[0], y, metrics['labels'][i], color=TEXT_MAIN, fontsize=12, va='center', transform=ax_table.transAxes)
        ax_table.text(cols_x[1], y, f"{metrics['mobile'][i]} {metrics['units'][i]}", color=colors_mobile[i], fontsize=13, fontweight='bold', va='center', transform=ax_table.transAxes)
        ax_table.text(cols_x[2], y, f"{metrics['resnet'][i]} {metrics['units'][i]}", color=colors_resnet[i], fontsize=13, fontweight='bold', va='center', transform=ax_table.transAxes)
        
        ax_table.add_patch(patches.FancyBboxPatch((cols_x[3]-0.02, y-0.04), 0.13, 0.08, boxstyle="round,pad=0.01,rounding_size=0.02", color=ratio_colors[i], transform=ax_table.transAxes, zorder=1))
        ax_table.text(cols_x[3]+0.045, y, ratio_texts[i], color=text_colors[i], fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_table.transAxes, zorder=2)

    fig2.savefig(RESULT_DIR / 'model_dashboard.png', bbox_inches='tight', pad_inches=0.2)
    print("✅ 'results/model_dashboard.png' 저장 완료!")

if __name__ == "__main__":
    main()