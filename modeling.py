# /content/drive/MyDrive/dataset/train_compare_baseline_dab_raw_cleanonly_2px_epochial_dump_scaled_fixed.py

import os
import math
import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# ---------- 1. 하이퍼파라미터 ----------
BATCH           = 128
EPOCHS          = 30
LR_WEIGHTS      = 1e-3       # Conv/FC weight 학습률
LR_ALPHA_SIGMA  = 1e-1       # AAReLU α 및 DABBlur σ 초기 학습률
DECAY_GAMMA     = 0.95       # 매 epoch마다 α/σ lr을 이 비율로 곱해 감소
SHIFT_PIX       = 2          # shift 픽셀 수 (고정)
NOISE_STD       = 25.0       # 노이즈 표준편차 (raw)

HEX_SAVE_ROOT   = "/content/drive/MyDrive/final"
os.makedirs(HEX_SAVE_ROOT, exist_ok=True)

# ---------- 2. 데이터 (EMNIST “balanced”, 47클래스; 28×28→Pad→32×32; RAW 입력) ----------
transform = transforms.Compose([
    transforms.Pad(2),                             # 28×28 → 32×32
    transforms.Lambda(lambda img: TF.hflip(img)),  # ① 좌우 반전
    transforms.Lambda(lambda img: TF.rotate(img, 90)),  # ② 90° 시계 방향 회전
    transforms.PILToTensor()                       # uint8 [0,255], 정규화 없음
])

train_ds = datasets.EMNIST(
    root="/content/drive/MyDrive/dataset", split="balanced",
    train=True, download=True,
    transform=transform
)
test_ds  = datasets.EMNIST(
    root="/content/drive/MyDrive/dataset", split="balanced",
    train=False, download=True,
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2)


# ---------- 3-1. Baseline LeNet-5 정의 ----------
class BaselineLeNet47(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0, bias=False)    # 32→28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)                           # 28→14

        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, bias=False)   # 14→10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)                           # 10→5

        self.conv3 = nn.Conv2d(16, 120, 5, padding=0, bias=False) # 5→1
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 47, bias=False)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))   # [B,6,14,14]
        x = self.pool2(self.relu2(self.conv2(x)))   # [B,16,5,5]
        x = self.relu3(self.conv3(x))               # [B,120,1,1]
        x = x.view(x.size(0), -1)                   # [B,120]
        x = self.relu4(self.fc1(x))                 # [B,84]
        x = self.fc2(x)                             # [B,47]
        return x


# ---------- 3-2. AA-ReLU 정의 (α 초기값 = 130, 80, 60 그대로) ----------
class AAReLU(nn.Module):
    def __init__(self, alpha_init=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.alpha, min=1e-4)
        cutoff = a * math.exp(math.pi / 2)
        y = torch.where(x > 0, x, torch.zeros_like(x))
        xs = torch.where(x < a, a, x)
        roll = a * torch.sin(torch.log(xs / a)) + a
        y = torch.where(x >= a, roll, y)
        y = torch.where(x >= cutoff, 2.0 * a, y)
        return y


# ---------- 3-3. DABBlur 정의 (σ 초기값 10, 10 그대로) ----------
def gaussian_kernel5(sigma):
    dev = sigma.device if isinstance(sigma, torch.Tensor) else torch.device("cpu")
    ax = torch.arange(-2, 3, device=dev, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))
    return kernel / kernel.sum()

class DABBlur5(nn.Module):
    def __init__(self, sigma_init=1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))

    def forward(self, x):
        k = gaussian_kernel5(torch.clamp(self.sigma, 1e-2, 10.0))
        k = k.to(x.device, x.dtype)[None, None, :, :]  # [1,1,5,5]
        x = nn.functional.conv2d(
            x,
            k.repeat(x.shape[1], 1, 1, 1),             # depthwise: [C,1,5,5]
            groups=x.shape[1],
            padding=2
        )
        return x


# ---------- 3-4. LeNet-5 + AAReLU + DABBlur + MaxPool 정의 ----------
class LeNetEMNIST_DABMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        # C1: α=130.0, σ1=10
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0, bias=False)
        self.act1  = AAReLU(alpha_init=130.0)
        self.dab1  = DABBlur5(sigma_init= 9.0)
        self.pool1 = nn.MaxPool2d(2, 2)    # 32→28→14

        # C2: α=80.0, σ2=10
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, bias=False)
        self.act2  = AAReLU(alpha_init=80.0)
        self.dab2  = DABBlur5(sigma_init= 9.0)
        self.pool2 = nn.MaxPool2d(2, 2)    # 14→10→5

        # C3: α=60.0
        self.conv3 = nn.Conv2d(16, 120, 5, padding=0, bias=False)
        self.act3  = AAReLU(alpha_init=60.0)

        self.fc1 = nn.Linear(120, 84, bias=False)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 47, bias=False)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.dab1(x)
        x = self.pool1(x)

        x = self.act2(self.conv2(x))
        x = self.dab2(x)
        x = self.pool2(x)

        x = self.act3(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- 4. Helper: shifted & noisy test sets (RAW, shift=2px 고정) ----------
def compute_shifted_accuracy(model, device, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, lab in loader:
            img = img.float().to(device)                        # RAW input
            img = torch.roll(img, shifts=(SHIFT_PIX, SHIFT_PIX), dims=(2, 3))
            lab = lab.to(device)
            pred = model(img).argmax(dim=1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
    return 100.0 * correct / total

def compute_noisy_accuracy(model, device, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, lab in loader:
            img = img.float().to(device)                        # RAW input
            noise = torch.randn_like(img) * NOISE_STD
            noisy = torch.clamp(img + noise, 0, 255)
            lab = lab.to(device)
            pred = model(noisy).argmax(dim=1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
    return 100.0 * correct / total


# ---------- 5. 모델 초기화 및 Optimizer ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

baseline = BaselineLeNet47().to(device)
opt_baseline = torch.optim.Adam(baseline.parameters(), lr=LR_WEIGHTS)

dabmodel = LeNetEMNIST_DABMaxPool().to(device)

# α/σ 파라미터만 별도 그룹으로 분리
alpha_sigma_params = []
weight_params = []
for name, param in dabmodel.named_parameters():
    if "alpha" in name or "sigma" in name:
        alpha_sigma_params.append(param)
    else:
        weight_params.append(param)

opt_dab = torch.optim.Adam([
    {"params": weight_params,      "lr": LR_WEIGHTS},
    {"params": alpha_sigma_params, "lr": LR_ALPHA_SIGMA}
])

criterion = nn.CrossEntropyLoss()


# ---------- 6. 학습 & 평가 루프 (Clean 데이터만 학습) ----------
for epoch in range(1, EPOCHS + 1):
    # --- (A) Baseline 학습 (Clean only) ---
    baseline.train()
    for img, lab in train_loader:
        img = img.float().to(device)  # RAW
        lab = lab.to(device)

        opt_baseline.zero_grad()
        logits = baseline(img)
        loss = criterion(logits, lab)
        loss.backward()
        opt_baseline.step()

    # --- (B) DAB 모델 학습 (Clean only) ---
    dabmodel.train()
    for img, lab in train_loader:
        img = img.float().to(device)  # RAW
        lab = lab.to(device)

        opt_dab.zero_grad()
        logits = dabmodel(img)
        loss = criterion(logits, lab)
        loss.backward()
        opt_dab.step()

    # --- (C) α/σ 학습률(decay) 적용 ---
    opt_dab.param_groups[1]['lr'] *= DECAY_GAMMA

    # --- (D) 평가: clean, shift, noise 모두 측정 ---
    baseline.eval()
    dabmodel.eval()

    # clean
    correct_b = total_b = 0
    correct_d = total_d = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img = img.float().to(device)  # RAW
            lab = lab.to(device)
            pred_b = baseline(img).argmax(dim=1)
            pred_d = dabmodel(img).argmax(dim=1)
            correct_b += (pred_b == lab).sum().item()
            correct_d += (pred_d == lab).sum().item()
            total_b += lab.size(0)
            total_d += lab.size(0)

    acc_clean_b = 100.0 * correct_b / total_b
    acc_clean_d = 100.0 * correct_d / total_d

    # shift = 2px
    acc_shift_b = compute_shifted_accuracy(baseline, device, test_loader)
    acc_shift_d = compute_shifted_accuracy(dabmodel, device, test_loader)

    # noise (σ=25)
    acc_noise_b = compute_noisy_accuracy(baseline, device, test_loader)
    acc_noise_d = compute_noisy_accuracy(dabmodel, device, test_loader)

    # --- (E) α & σ 및 학습률 현황 출력 ---
    a1  = dabmodel.act1.alpha.detach().item()
    a2  = dabmodel.act2.alpha.detach().item()
    a3  = dabmodel.act3.alpha.detach().item()
    s1  = dabmodel.dab1.sigma.detach().item()
    s2  = dabmodel.dab2.sigma.detach().item()
    lr_a_sigma = opt_dab.param_groups[1]['lr']

    print(f"[Epoch {epoch}]")
    print(f"  Baseline: Clean={acc_clean_b:.2f}%, Shift=({SHIFT_PIX}px)={acc_shift_b:.2f}%, Noise={acc_noise_b:.2f}%")
    print(f"       DAB: Clean={acc_clean_d:.2f}%, Shift=({SHIFT_PIX}px)={acc_shift_d:.2f}%, Noise={acc_noise_d:.2f}%")
    print(f"    Alpha: a1={a1:.4f}, a2={a2:.4f}, a3={a3:.4f}")
    print(f"    Sigma: σ1={s1:.4f}, σ2={s2:.4f} |  lr_α/σ={lr_a_sigma:.5f}\n")

    # --- (F) Epoch별 HEX 덤프 디렉토리 생성 ---
    epoch_dir = os.path.join(HEX_SAVE_ROOT, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # ── (F-1) Conv1 weight [6,1,5,5] RAW 그대로, scale = 127/max_abs ──
    w1 = dabmodel.conv1.weight.clone().cpu().float()
    max_abs_w1 = torch.max(torch.abs(w1))
    scale_w1 = 127.0 / max_abs_w1 if max_abs_w1 > 0 else 1.0
    w1_int8 = torch.round(w1 * scale_w1).clamp(-128, 127).to(torch.int8)
    arr = w1_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "out_conv1_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-2) Conv2 weight [16,6,5,5] RAW 그대로, scale = 127/max_abs ──
    w2 = dabmodel.conv2.weight.clone().cpu().float()
    max_abs_w2 = torch.max(torch.abs(w2))
    scale_w2 = 127.0 / max_abs_w2 if max_abs_w2 > 0 else 1.0
    w2_int8 = torch.round(w2 * scale_w2).clamp(-128, 127).to(torch.int8)
    arr = w2_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "out_conv2_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-3) Conv3 weight [120,16,5,5] RAW 그대로, scale = 127/max_abs ──
    w3 = dabmodel.conv3.weight.clone().cpu().float()
    max_abs_w3 = torch.max(torch.abs(w3))
    scale_w3 = 127.0 / max_abs_w3 if max_abs_w3 > 0 else 1.0
    w3_int8 = torch.round(w3 * scale_w3).clamp(-128, 127).to(torch.int8)
    arr = w3_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "out_conv3_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-4) FC1 weight [84,120] RAW 그대로, scale = 127/max_abs ──
    fc1_w = dabmodel.fc1.weight.clone().cpu().float()
    max_abs_fc1 = torch.max(torch.abs(fc1_w))
    scale_fc1 = 127.0 / max_abs_fc1 if max_abs_fc1 > 0 else 1.0
    fc1_int8 = torch.round(fc1_w * scale_fc1).clamp(-128, 127).to(torch.int8)
    arr = fc1_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "out_fc1_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-5) FC2 weight [47,84] RAW 그대로, scale = 127/max_abs ──
    fc2_w = dabmodel.fc2.weight.clone().cpu().float()
    max_abs_fc2 = torch.max(torch.abs(fc2_w))
    scale_fc2 = 127.0 / max_abs_fc2 if max_abs_fc2 > 0 else 1.0
    fc2_int8 = torch.round(fc2_w * scale_fc2).clamp(-128, 127).to(torch.int8)
    arr = fc2_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "out_fc2_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-6) Blur1 kernel [5,5] → [6,1,5,5]로 반복, scale = 127 ──
    sigma1 = dabmodel.dab1.sigma.detach().cpu().item()
    kernel1 = gaussian_kernel5(torch.tensor(sigma1)).cpu()  # [5,5]
    blur1 = kernel1.unsqueeze(0).unsqueeze(0).repeat(6, 1, 1, 1)  # [6,1,5,5]
    blur1_int8 = torch.round(blur1 * 127.0).clamp(-128, 127).to(torch.int8)
    arr = blur1_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "blur1_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-7) Blur2 kernel [5,5] → [16,6,5,5]로 반복, scale = 127 ──
    sigma2 = dabmodel.dab2.sigma.detach().cpu().item()
    kernel2 = gaussian_kernel5(torch.tensor(sigma2)).cpu()  # [5,5]
    blur2 = kernel2.unsqueeze(0).unsqueeze(0).repeat(16, 6, 1, 1)  # [16,6,5,5]
    blur2_int8 = torch.round(blur2 * 127.0).clamp(-128, 127).to(torch.int8)
    arr = blur2_int8.view(-1).cpu().numpy().astype("uint8")
    if arr.size % 4 != 0:
        pad = 4 - (arr.size % 4)
        arr = np.concatenate([arr, np.zeros(pad, dtype="uint8")])
    words = []
    for i in range(0, arr.size, 4):
        b0, b1, b2, b3 = arr[i : i + 4]
        word = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8) | int(b3)
        words.append(f"{word:08x}")
    with open(os.path.join(epoch_dir, "blur2_32.hex"), "w") as f:
        f.write("\n".join(words))

    # ── (F-8) α/σ 값 TXT로 저장 ──
    save_path = os.path.join(epoch_dir, "alpha_sigma_values.txt")
    with open(save_path, "w") as f:
        f.write(f"alpha1 = {dabmodel.act1.alpha.item():.6f}\n")
        f.write(f"alpha2 = {dabmodel.act2.alpha.item():.6f}\n")
        f.write(f"alpha3 = {dabmodel.act3.alpha.item():.6f}\n")
        f.write(f"sigma1 = {dabmodel.dab1.sigma.item():.6f}\n")
        f.write(f"sigma2 = {dabmodel.dab2.sigma.item():.6f}\n")

    print(f"▶ Epoch {epoch} hex files & alpha/sigma values saved to {epoch_dir}\n")

print("▶ Training complete.")
