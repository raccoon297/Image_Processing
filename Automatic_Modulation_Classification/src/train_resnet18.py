import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "RML2016.10a_dict.pkl"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODEL_DIR / "resnet18_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "resnet18_final.pth"
CLASSES_PATH = MODEL_DIR / "classes.pkl"
RESULT_IMAGE_PATH = RESULT_DIR / "resnet18_training_result.png"
METRICS_PATH = RESULT_DIR / "resnet18_metrics.json"

# ============================================================
# 1. 전처리: IQ 샘플 → 32x32 성상도 이미지
# ============================================================
def iq_to_constellation(iq_sample, grid_size=32, extent=1.5):
    """
    AGC(자동 이득 제어) 정규화 후 IQ 좌표를 2D 밀도 히스토그램으로 변환.
    - max_val 정규화: SNR이 낮아 원점에 뭉친 신호를 흩뿌림
    - gaussian_filter(sigma=0.3): 1~2픽셀짜리 점을 CNN이 인식하도록 아주 옅게 번짐
    """
    max_val = np.max(np.abs(iq_sample))
    if max_val > 0:
        iq_sample = iq_sample / max_val

    i_data = iq_sample[0, :]
    q_data = iq_sample[1, :]

    img, _, _ = np.histogram2d(
        i_data, q_data,
        bins=grid_size,
        range=[[-extent, extent], [-extent, extent]]
    )

    img = img.T
    img = gaussian_filter(img, sigma=0.3)

    if img.max() > 0:
        img = img / img.max()

    return img


# ============================================================
# 2. 데이터셋 클래스
#    - SNR 정보를 함께 보관 → SNR별 정확도 평가에 활용
# ============================================================
class RadioMLDataset(Dataset):
    def __init__(self, filepath, snr_threshold=10):
        with open(filepath, 'rb') as f:
            xd = pickle.load(f, encoding='latin1')

        self.data, self.labels, self.snrs = [], [], []
        self.target_classes = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64']
        self.classes = sorted(self.target_classes)
        self.class_to_idx = {mod: idx for idx, mod in enumerate(self.classes)}

        for key in xd.keys():
            mod_type, snr = key
            if snr >= snr_threshold and mod_type in self.target_classes:
                for iq_sample in xd[key]:
                    self.data.append(iq_sample)
                    self.labels.append(self.class_to_idx[mod_type])
                    self.snrs.append(snr)           # ★ SNR 값 저장

        self.data   = np.array(self.data)
        self.labels = np.array(self.labels)
        self.snrs   = np.array(self.snrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iq_sample  = self.data[idx]
        img_matrix = iq_to_constellation(iq_sample)
        img_tensor = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0)  # (1, 32, 32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor


# ============================================================
# 3. CNN 아키텍처: ResNet-18 기반 (32×32 그레이스케일 최적화)
# ============================================================
class ConstellationCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ConstellationCNN, self).__init__()

        # PyTorch 공식 ResNet-18 로드 (사전학습 가중치 없이 랜덤 초기화)
        self.resnet = models.resnet18(weights=None)

        # ── [정문 개조] ──────────────────────────────────────
        # 원본 ResNet-18의 첫 Conv: kernel=7, stride=2, padding=3 (224×224 기준 설계)
        # 32×32 입력에서 stride=2를 그대로 쓰면 16×16으로 즉시 줄어들어 정보 손실이 큼.
        # → kernel=3, stride=1, padding=1로 교체해 feature map을 32×32로 유지.
        # → in_channels=1: RGB 3채널 → 그레이스케일 성상도 이미지 1채널로 변경.
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # ── [MaxPool 제거] ───────────────────────────────────
        # 원본 ResNet은 첫 Conv 직후 MaxPool(kernel=3, stride=2)로 한 번 더 다운샘플링.
        # 32×32에서 이걸 그대로 두면 16×16이 돼버림 → Identity로 교체해 건너뜀.
        self.resnet.maxpool = nn.Identity()

        # ── [후문 개조] ──────────────────────────────────────
        # ResNet-18 기본 fc 출력: 1000 (ImageNet 클래스 수)
        # → 우리 타겟 6클래스(BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64)로 교체.
        in_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),           # 과적합 방지용 Dropout 삽입
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# ============================================================
# 4. 유틸리티: 검증 루프
# ============================================================
def evaluate(model, loader, criterion, device):
    """val/test 데이터에 대한 loss·accuracy 반환"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============================================================
# 5. SNR별 정확도 평가
# ============================================================
def evaluate_by_snr(model, full_dataset, eval_subset, device, batch_size=256):
    """검증 서브셋만 SNR별로 분리하여 정확도를 계산한다.

    eval_subset은 random_split으로 생성된 Subset 객체이며,
    eval_subset.indices에는 full_dataset 기준 원본 인덱스가 저장되어 있다.
    """
    model.eval()

    eval_indices = np.asarray(eval_subset.indices)
    eval_snrs = full_dataset.snrs[eval_indices]
    unique_snrs = sorted(set(eval_snrs.tolist()))
    snr_acc = {}

    for snr in unique_snrs:
        snr_indices = eval_indices[eval_snrs == snr]
        snr_subset = torch.utils.data.Subset(full_dataset, snr_indices.tolist())
        loader = DataLoader(snr_subset, batch_size=batch_size, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        snr_acc[snr] = 100.0 * correct / total

    return snr_acc


# ============================================================
# 6. 혼동행렬 수집용 예측값 수집
# ============================================================
def collect_predictions(model, loader, device):
    """검증 데이터 전체에 대한 (정답, 예측) 리스트 반환"""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


# ============================================================
# 7. 시각화: 학습 결과 통합 대시보드
#    ┌────────────┬────────────┐
#    │  Loss 곡선 │  Acc  곡선 │
#    ├────────────┴────────────┤
#    │      SNR별 정확도        │
#    ├─────────────────────────┤
#    │       혼동행렬           │
#    └─────────────────────────┘
# ============================================================
def plot_results(history, snr_acc, all_labels, all_preds, class_names, epochs):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 18), dpi=130)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    epochs_range = range(1, epochs + 1)
    COLOR_TRAIN = '#FF4B4B'
    COLOR_VAL   = '#FF9800'
    COLOR_SNR   = '#1E88E5'

    # ── (1) Loss 곡선 ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss',
             color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs_range, history['val_loss'],   label='Val Loss',
             color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=4, linestyle='--')
    ax1.set_title('Loss over Epochs', fontsize=15, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ── (2) Accuracy 곡선 ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc',
             color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=4)
    ax2.plot(epochs_range, history['val_acc'],   label='Val Acc',
             color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=4, linestyle='--')
    ax2.set_title('Accuracy over Epochs', fontsize=15, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # ── (3) SNR별 정확도 ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    snr_vals = list(snr_acc.keys())
    acc_vals = list(snr_acc.values())
    bars = ax3.bar(snr_vals, acc_vals, color=COLOR_SNR, alpha=0.85, width=1.4, edgecolor='white')
    ax3.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, label='80% Baseline')
    ax3.axhline(y=95, color='green',  linestyle='--', linewidth=1.5, label='95% Baseline')

    # 막대 위에 수치 표시
    for bar, acc in zip(bars, acc_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_title('Accuracy by SNR Level (Validation Set)', fontsize=15, fontweight='bold', pad=12)
    ax3.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 110])
    ax3.set_xticks(snr_vals)
    ax3.legend(fontsize=11)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)

    # ── (4) 혼동행렬 ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax4, cmap='Blues', colorbar=False, xticks_rotation=30)
    ax4.set_title('Confusion Matrix (Validation Set)', fontsize=15, fontweight='bold', pad=12)
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')

    plt.savefig(RESULT_IMAGE_PATH, bbox_inches='tight')
    print(f"\n📊 '{RESULT_IMAGE_PATH}' 저장 완료 (Loss / Acc / SNR별 정확도 / 혼동행렬 통합)")


# ============================================================
# 8. 메인 학습 함수
# ============================================================
def train():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # ── 하이퍼파라미터 ─────────────────────────────────────
    filepath      = DATA_PATH
    batch_size    = 128
    epochs        = 30
    lr            = 0.001
    val_ratio     = 0.2          # 검증셋 비율 20%
    snr_threshold = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 데이터 로드 & Train/Val 분리 ───────────────────────
    print("📂 데이터셋 로딩 및 전처리 중...")
    full_dataset = RadioMLDataset(filepath, snr_threshold=snr_threshold)

    val_size   = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)   # 재현성 보장
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  전체: {len(full_dataset):,}  |  학습: {train_size:,}  |  검증: {val_size:,}")

    # ── 모델 / 손실함수 / 옵티마이저 ───────────────────────
    model     = ConstellationCNN(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    # weight_decay=1e-4: L2 정규화로 가중치가 너무 커지는 것을 억제 → 과적합 방지
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ★ 개선 3: ReduceLROnPlateau 스케줄러
    #   val_loss가 patience(5에포크) 동안 개선 없으면 lr × factor(0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── 히스토리 초기화 ────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    best_val_epoch = 0
    # Early Stopping: val_loss가 patience 에포크 동안 개선 없으면 학습 조기 종료
    patience       = 7
    no_improve     = 0

    print(f"\n🚀 [{device}] 학습 시작")
    print(f"   클래스: {full_dataset.classes}")
    print(f"   에포크: {epochs}  |  배치: {batch_size}  |  초기 lr: {lr}\n")

    for epoch in range(epochs):
        # ── 학습 단계 ──────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc':  f'{100*correct/total:.2f}%'})

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total

        # ── 검증 단계 ──────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ── 스케줄러 업데이트 (val_loss 기준) ──────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 최고 val_loss 모델 저장 + Early Stopping 카운터 관리
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            no_improve     = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"lr: {current_lr:.6f} | No improve: {no_improve}/{patience}")

        # Early Stopping 발동
        if no_improve >= patience:
            print(f"\n[Early Stopping] Val Loss가 {patience}에포크 동안 개선되지 않아 학습을 조기 종료합니다.")
            print(f"  최고 성능 에포크: {best_val_epoch} / Val Loss: {best_val_loss:.4f}")
            break

    print(f"\n✅ 학습 완료!  최고 Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})")

    # ── 최고 모델 로드 후 최종 평가 ────────────────────────
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    best_eval_loss, best_eval_acc = evaluate(model, val_loader, criterion, device)
    print(f"최고 체크포인트 검증 정확도: {best_eval_acc:.2f}%")
    print("\n📐 SNR별 정확도 계산 중...")
    snr_acc = evaluate_by_snr(model, full_dataset, val_dataset, device)

    print("\n[SNR별 정확도]")
    for snr, acc in sorted(snr_acc.items()):
        bar = '█' * int(acc / 5)
        print(f"  SNR {snr:+3d} dB | {bar:<20s} {acc:.1f}%")

    # ── 혼동행렬용 예측값 수집 (검증셋 기준) ───────────────
    print("\n🔢 혼동행렬 계산 중...")
    all_labels, all_preds = collect_predictions(model, val_loader, device)

    # ── 통합 시각화 ────────────────────────────────────────
    plot_results(history, snr_acc, all_labels, all_preds,
                 full_dataset.classes, len(history['train_loss']))

    # ── 모델 저장 ──────────────────────────────────────────
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(full_dataset.classes, f)

    metrics = {
        'model_name': 'ResNet-18',
        'best_epoch': best_val_epoch,
        'best_val_loss': float(best_eval_loss),
        'best_val_accuracy': float(best_eval_acc),
        'mean_val_snr_accuracy': float(np.mean(list(snr_acc.values()))),
        'val_snr_accuracy': {str(k): float(v) for k, v in snr_acc.items()},
        'snr_evaluation_scope': 'validation_set',
        'num_train_samples': train_size,
        'num_val_samples': val_size,
        'snr_threshold': snr_threshold,
    }
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"💾 최종 가중치 '{FINAL_MODEL_PATH}' 및 클래스 정보 '{CLASSES_PATH}' 저장 완료.")
    print(f"📄 학습 지표 '{METRICS_PATH}' 저장 완료.")


if __name__ == "__main__":
    train()