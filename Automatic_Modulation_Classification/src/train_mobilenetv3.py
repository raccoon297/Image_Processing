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
BEST_MODEL_PATH = MODEL_DIR / "mobilenetv3_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "mobilenetv3_final.pth"
CLASSES_PATH = MODEL_DIR / "classes.pkl"
RESULT_IMAGE_PATH = RESULT_DIR / "mobilenetv3_training_result.png"
METRICS_PATH = RESULT_DIR / "mobilenetv3_metrics.json"

# ============================================================
# 1. 전처리: IQ 샘플 → 32x32 성상도 이미지
# ============================================================
def iq_to_constellation(iq_sample, grid_size=32, extent=1.5):
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
                    self.snrs.append(snr)

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
# 3. CNN 아키텍처: MobileNetV3 (Small) 기반 과적합 방지 세팅
# ============================================================
class ConstellationCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(ConstellationCNN, self).__init__()
        
        self.mobilenet = models.mobilenet_v3_small(weights=None)
        
        # [정문 개조] 1채널 입력에 맞게 수정
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # [과적합 방지 추가] 기존 내장 Dropout(p=0.2)의 비율을 0.5로 상향하여 규제 강화
        self.mobilenet.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        
        # [후문 개조] 출력 클래스 개수 조정 (6개)
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


# ============================================================
# 4. 유틸리티 함수들
# ============================================================
def evaluate(model, loader, criterion, device):
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


def collect_predictions(model, loader, device):
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
# 5. 시각화 대시보드 (학습 중단 시점까지의 유연한 에포크 축 처리 적용)
# ============================================================
def plot_results(history, snr_acc, all_labels, all_preds, class_names):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 18), dpi=130)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 실제 진행된 에포크 수에 맞게 range 설정
    actual_epochs = len(history['train_loss'])
    epochs_range = range(1, actual_epochs + 1)
    
    COLOR_TRAIN = '#FF4B4B'
    COLOR_VAL   = '#FF9800'
    COLOR_SNR   = '#1E88E5'

    # (1) Loss 곡선
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs_range, history['val_loss'],   label='Val Loss', color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=4, linestyle='--')
    ax1.set_title('Loss over Epochs', fontsize=15, fontweight='bold', pad=12)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # (2) Accuracy 곡선
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc', color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=4)
    ax2.plot(epochs_range, history['val_acc'],   label='Val Acc', color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=4, linestyle='--')
    ax2.set_title('Accuracy over Epochs', fontsize=15, fontweight='bold', pad=12)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # (3) SNR별 정확도
    ax3 = fig.add_subplot(gs[1, :])
    snr_vals = list(snr_acc.keys())
    acc_vals = list(snr_acc.values())
    bars = ax3.bar(snr_vals, acc_vals, color=COLOR_SNR, alpha=0.85, width=1.4, edgecolor='white')
    ax3.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, label='80% Baseline')
    ax3.axhline(y=95, color='green',  linestyle='--', linewidth=1.5, label='95% Baseline')

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

    # (4) 혼동행렬
    ax4 = fig.add_subplot(gs[2, :])
    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax4, cmap='Blues', colorbar=False, xticks_rotation=30)
    ax4.set_title('Confusion Matrix (Validation Set)', fontsize=15, fontweight='bold', pad=12)
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')

    plt.savefig(RESULT_IMAGE_PATH, bbox_inches='tight')
    print(f"\n'{RESULT_IMAGE_PATH}' 저장 완료 (Loss / Acc / SNR별 정확도 / 혼동행렬)")


# ============================================================
# 6. 메인 학습 함수 (Early Stopping 및 Weight Decay 적용)
# ============================================================
def train():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    filepath      = DATA_PATH
    batch_size    = 128
    epochs        = 30
    lr            = 0.001
    val_ratio     = 0.2
    snr_threshold = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("데이터셋 로딩 및 전처리 중...")
    full_dataset = RadioMLDataset(filepath, snr_threshold=snr_threshold)

    val_size   = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  전체: {len(full_dataset):,}  |  학습: {train_size:,}  |  검증: {val_size:,}")

    model     = ConstellationCNN(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # [과적합 방지 추가] weight_decay=1e-4 옵션을 주어 가중치 감쇠(L2 Regularization) 활성화
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3 # 에포크 단축에 발맞춰 체감 주기 최적화
    )

    # [과적합 방지 추가] 조기 종료(Early Stopping) 관련 변수 선언
    early_stop_patience = 5
    early_stop_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    best_val_epoch = 0

    print(f"\n[{device}] 학습 시작")
    print(f"   클래스: {full_dataset.classes}")
    print(f"   최대 에포크: {epochs}  |  배치: {batch_size}  |  초기 lr: {lr}\n")

    for epoch in range(epochs):
        # ── 학습 단계 ──
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc':  f'{100*correct/total:.2f}%'})

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total

        # ── 검증 단계 ──
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
              f"lr: {current_lr:.6f}")

        # 최고 성능 갱신 여부 확인 및 조기 종료 카운트 계산
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            early_stop_counter = 0  # 개선되었으므로 카운터 초기화
        else:
            early_stop_counter += 1 # 개선되지 않았으므로 카운트 누적
            
        # [과적합 방지 추가] 지정된 횟수동안 개선되지 않으면 강제 종료
        if early_stop_counter >= early_stop_patience:
            print(f"\n[Early Stopping] {early_stop_patience} 에포크 동안 Val Loss가 개선되지 않아 학습을 조기 종료합니다.")
            break

    print(f"\n학습 완료! 최고 Val Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})")

    # ── 최고 가중치 로드 후 최종 평가 ──
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    best_eval_loss, best_eval_acc = evaluate(model, val_loader, criterion, device)
    print(f"최고 체크포인트 검증 정확도: {best_eval_acc:.2f}%")
    print("\nSNR별 정확도 계산 중...")
    snr_acc = evaluate_by_snr(model, full_dataset, val_dataset, device)

    print("\n[SNR별 정확도]")
    for snr, acc in sorted(snr_acc.items()):
        bar = '█' * int(acc / 5)
        print(f"   SNR {snr:+3d} dB | {bar:<20s} {acc:.1f}%")

    print("\n혼동행렬 계산 중...")
    all_labels, all_preds = collect_predictions(model, val_loader, device)

    # 통합 시각화 결과물 출력
    plot_results(history, snr_acc, all_labels, all_preds, full_dataset.classes)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(full_dataset.classes, f)

    metrics = {
        'model_name': 'MobileNetV3 Small',
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