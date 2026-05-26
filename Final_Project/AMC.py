import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# --- 1. 전처리: 해상도를 32x32로 낮춰 밀도를 높이고, 아주 옅은 블러만 적용 ---
def iq_to_constellation(iq_sample, grid_size=32, extent=1.5):
    # [영상신호처리 & 통신 핵심 포인트: AGC(자동 이득 제어) 정규화]
    # 모든 신호의 최대 진폭을 1.0으로 강제로 쫙 늘려줍니다. (원점에 뭉친 점들을 흩뿌림)
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
    img = gaussian_filter(img, sigma=0.5)
    
    if img.max() > 0:
        img = img / img.max()
        
    return img

class RadioMLDataset(Dataset):
    def __init__(self, filepath, snr_threshold=10):
        with open(filepath, 'rb') as f:
            xd = pickle.load(f, encoding='latin1')
            
        self.data, self.labels = [], []
        self.target_classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
        self.classes = sorted(self.target_classes)
        self.class_to_idx = {mod: idx for idx, mod in enumerate(self.classes)}
        
        for key in xd.keys():
            mod_type, snr = key
            if snr >= snr_threshold and mod_type in self.target_classes:
                for iq_sample in xd[key]:
                    self.data.append(iq_sample)
                    self.labels.append(self.class_to_idx[mod_type])
                    
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iq_sample = self.data[idx]
        img_matrix = iq_to_constellation(iq_sample)
        # 1채널 텐서로 변환 (1, 32, 32)
        img_tensor = torch.tensor(img_matrix, dtype=torch.float32).unsqueeze(0) 
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor

# --- 2. CNN 아키텍처: 입력 크기 32x32에 맞춰 계산식 수정 ---
class ConstellationCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ConstellationCNN, self).__init__()
        # 입력 1채널 유지
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 8x8 -> 4x4
        
        # Flatten 크기 수정 (128채널 * 4 * 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train():
    filepath = 'RML2016.10a_dict.pkl'
    batch_size = 128
    epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("데이터셋 로딩 및 전처리 중...")
    dataset = RadioMLDataset(filepath, snr_threshold=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ConstellationCNN(num_classes=len(dataset.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    history_loss, history_acc = [], []
    
    print(f"\n[{device}] 학습 시작 (총 데이터: {len(dataset)}개)")
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100 * correct / total:.2f}%"})
            
        history_loss.append(running_loss / len(dataloader))
        history_acc.append(100 * correct / total)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='Train Loss', color='red')
    plt.title('Loss History')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_acc, label='Train Accuracy', color='blue')
    plt.title('Accuracy History')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_result.png')
    
    torch.save(model.state_dict(), 'amc_model.pth')
    with open('amc_classes.pkl', 'wb') as f:
        pickle.dump(dataset.classes, f)
    print("저장 완료.")

if __name__ == "__main__":
    train()