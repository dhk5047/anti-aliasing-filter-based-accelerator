import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torchvision.transforms as transforms
import time

# -------------------------------
# DABPool Layer (Depth Adaptive Blur Pooling)
# -------------------------------
class DABPool(nn.Module):
    def __init__(self, channels, sigma_init=1.0):
        super(DABPool, self).__init__()
        self.channels = channels
        self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))
        coords = torch.tensor([-1.0, 0.0, 1.0])
        self.register_buffer('grid_x', coords.view(1, 3).repeat(3, 1))
        self.register_buffer('grid_y', coords.view(3, 1).repeat(1, 3))

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=1e-6)
        g = torch.exp(-(self.grid_x**2 + self.grid_y**2) / (2 * sigma**2))
        kernel = g / g.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1)
        out = F.conv2d(x, kernel, bias=None, stride=2, padding=1, groups=self.channels)
        return out

# -------------------------------
# AAReLU Activation (Anti-Aliasing ReLU)
# -------------------------------
class AAReLU(nn.Module):
    def __init__(self, init_alpha=6.0):
        super(AAReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        a = torch.clamp(self.alpha, min=1e-6)
        cutoff = a * math.exp(math.pi / 2)
        y = torch.where(x > 0, x, torch.zeros_like(x))
        x_safe = torch.where(x < a, a, x)
        roll_val = a * torch.sin(torch.log(x_safe / a)) + a
        y = torch.where(x >= a, roll_val, y)
        y = torch.where(x >= cutoff, 2 * a, y)
        return y

# -------------------------------
# Bottleneck Block with AA-ReLU and optional DABPool
# -------------------------------
class Bottleneck_AA(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, blurpool_main=None):
        super(Bottleneck_AA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck_AA.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * Bottleneck_AA.expansion)
        self.act1  = AAReLU()
        self.act2  = AAReLU()
        self.act3  = AAReLU()
        self.blurpool_main = blurpool_main
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.blurpool_main is not None:
            out = self.blurpool_main(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out

# -------------------------------
# ResNet-101 with DABPool + AAReLU (Anti-Aliasing ResNet-101)
# -------------------------------
class ResNet101_AntiAliased(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101_AntiAliased, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.act1  = AAReLU()
        self.dense_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.blurpool1 = DABPool(channels=64, sigma_init=0.5)
        self.blurpool2 = DABPool(channels=64, sigma_init=1.0)

        self.inplanes = 64
        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=23, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        downsample = None
        blur_main = None
        if stride != 1 or self.inplanes != planes * 4:
            down_conv = nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=1, bias=False)
            down_bn   = nn.BatchNorm2d(planes * 4)
            if planes == 128:
                sigma_init = 1.5
            elif planes == 256:
                sigma_init = 2.0
            elif planes == 512:
                sigma_init = 2.5
            else:
                sigma_init = 1.5
            blur_main = DABPool(channels=planes, sigma_init=sigma_init)
            blur_skip = DABPool(channels=planes * 4, sigma_init=sigma_init)
            downsample = nn.Sequential(down_conv, down_bn, blur_skip)
        layers.append(Bottleneck_AA(self.inplanes, planes, stride=stride, downsample=downsample, blurpool_main=blur_main))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck_AA(self.inplanes, planes, stride=1, downsample=None, blurpool_main=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dense_maxpool(out)
        out = self.blurpool1(out)
        out = self.dense_maxpool(out)
        out = self.blurpool2(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# -------------------------------
# Utility: Diagonal Shift (이미지를 대각선으로 이동)
# -------------------------------
def apply_diagonal_shift(images, pixels=5):
    single_image = False
    if images.dim() == 3:
        images = images.unsqueeze(0)
        single_image = True
    N, C, H, W = images.shape
    padded = F.pad(images, (pixels, 0, pixels, 0))
    shifted = padded[:, :, 0:H, 0:W]
    return shifted if not single_image else shifted.squeeze(0)

# -------------------------------
# Main Training Loop with Debug Logging
# -------------------------------
if __name__ == '__main__':
    overall_start = time.time()
    print("=== 학습 시작 ===")
    
    # 데이터셋 준비 (CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    model = ResNet101_AntiAliased(num_classes=10).to(device)
    print("모델 초기화 완료.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 8], gamma=0.2)  # 에폭 수가 적으므로 milestones 조정

    epochs = 10  # 테스트용 에폭 수
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        print(f"\n=== Epoch {epoch+1}/{epochs} 시작 ===")
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)
            
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            
            # 배치 진행 정보 출력 (50 배치마다)
            if batch_idx % 50 == 0:
                batch_time = time.time() - batch_start
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}] -> Loss: {loss.item():.4f} (배치 처리 시간: {batch_time:.2f} sec)")
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / batch_count
        print(f"--- Epoch {epoch+1} 종료 ---")
        print(f"에폭 평균 Loss: {epoch_loss:.4f}, 에폭 소요 시간: {epoch_time:.2f} sec")
        
        # Test set 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")
    
    overall_time = time.time() - overall_start
    print(f"\n=== 학습 완료: 전체 소요 시간 {overall_time:.2f} sec ===")

    # Diagonal Shift 예시 출력
    data_iter = iter(test_loader)
    example_images, _ = next(data_iter)
    shifted_images = apply_diagonal_shift(example_images, pixels=3)
    print("대각선 shift 함수가 적용되었습니다. (예시 이미지 배치)")
