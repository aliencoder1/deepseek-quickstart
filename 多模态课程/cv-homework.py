# encoding=utf-8

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

# 全局参数
DATA_ROOT = '/Users/emilia/Documents/GitHub/deepseek-quickstart/多模态课程/CV-车辆检测'
IMAGE_FOLDER = os.path.join(DATA_ROOT, 'image')
TRAIN_TXT = os.path.join(DATA_ROOT, 're_id_1000_train.txt')
TEST_TXT = os.path.join(DATA_ROOT, 're_id_1000_test.txt')

BATCH_SIZE = 48
LR = 0.00035
EPOCHS = 40
NUM_CLASSES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 8  # 用于Early Stopping，当验证集准确率连续Patience轮不上升时停止训练

# 车辆重识别数据集加载类(继承自torch.utils.data.Dataset)
class VehicleDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        # :param txt_file: 包含图片路径和标签的txt文件路径
        # :param root_dir: 图片存放的根目录
        # :param transform: 预处理操作

        self.root_dir = root_dir
        self.transform = transform
        self.img_infos = []

        # 读取txt文件并解析路径和标签(格式示例:1/License_1/440400.jpg)
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 统一使用正斜杠处理路径，避免Windows/MacOS/Linux路径兼容问题
                    rel_path = line.replace('\\', '/')

                    try:
                        # 解析(路径第二部分为License_XXX)
                        parts = rel_path.split('/')
                        if len(parts) >= 2:
                            license_tag = parts[1]
                            # 提取数字ID并转为0-999的索引(原始数据是从1开始的)
                            label_id = int(license_tag.split('_')[1]) - 1
                            self.img_infos.append((rel_path, label_id))
                    except (IndexError, ValueError):
                        # 如果某行数据格式异常，则跳过
                        continue
        else:
            print(f"Warning: File {txt_file} not found.")

    def __len__(self):
        # 返回数据集大小
        return len(self.img_infos)

    def __getitem__(self, idx):
        # 根据索引获取单条数据
        rel_path, label = self.img_infos[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # 使用RGB模式读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 工程容错：如果图片损坏，返回一张全黑图片防止训练中断
            image = Image.new('RGB', (320, 320))

        # 执行预处理转换
        if self.transform:
            image = self.transform(image)
        return image, label

# 基于ResNet50的车辆Re-ID模型
class ReIDResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ReIDResNet50, self).__init__()

        # 加载ImageNet预训练权重，加速收敛
        weights = ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights=weights)

        # 原生ResNet50总步长为32，移除了最后一个stage的下采样，使特征图尺寸变大一倍（保留更高的特征图分辨率）；
        base_model.layer4[0].downsample[0].stride = (1, 1)
        base_model.layer4[0].conv2.stride = (1, 1)

        # 提取除最后全连接层之外的所有层
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        # 自适应平均池化，将特征图大小固定为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # BNNeck结构：在特征层和分类层之间加入BN层（有助于特征的收敛）
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)  # 冻结偏置项，避免过拟合

        # 最终分类器，不使用偏置项
        self.classifier = nn.Linear(2048, num_classes, bias=False)

        # 初始化分类器权重，使用正态分布
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # 前向传播过程
        x = self.backbone(x)
        x = self.avgpool(x)

        # 展平特征向量
        x = x.view(x.size(0), -1)

        # 通过BNNeck
        feat = self.bn(x)

        # 通过分类器得到Logits
        x = self.classifier(feat)
        return x

# 定义数据增强和预处理策略
def get_transforms():
    # :return: 包含train和val转换的字典
    return {
        'train': transforms.Compose([
            # 将图片统一Resize到320x320，比标准的224x224更大，以保留细节
            transforms.Resize((320, 320)),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(),
            # Padding后随机裁剪，增加位置扰动
            transforms.Pad(10),
            transforms.RandomCrop(320),
            transforms.ToTensor(),
            # ImageNet标准归一化参数
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # 随机擦除 (Random Erasing)，强迫模型关注局部特征而非整体
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
        ]),
        'val': transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# 主函数包含：模型构建、数据加载、训练循环、验证及模型保存
def main():
    print(f"Current Device: {DEVICE}")

    # 获取数据预处理策略
    data_transforms = get_transforms()

    # 实例化数据集
    train_dataset = VehicleDataset(TRAIN_TXT, IMAGE_FOLDER, transform=data_transforms['train'])
    test_dataset = VehicleDataset(TEST_TXT, IMAGE_FOLDER, transform=data_transforms['val'])

    # 实例化DataLoader
    # num_workers根据CPU核心数设定，pin_memory加速GPU数据传输
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(test_dataset)}")

    # 初始化模型并移动到设备
    model = ReIDResNet50(NUM_CLASSES)
    model = model.to(DEVICE)

    # 定义损失函数：使用Label Smoothing(0.1)来防止过拟合，防止模型对单一标签过于自信
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 定义优化器，使用Adam算法，初始学习率设为3.5e-4
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    # 学习率调度器，使用余弦退火策略让学习率平滑下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环变量初始化
    best_acc = 0.0
    early_stop_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')

        # 分别进行训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 启用Dropout和BN更新
                dataloader = train_loader
            else:
                model.eval()  # 冻结BN并禁用Dropout
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # 批次循环
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播（只有训练阶段才追踪梯度）
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计Loss和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 更新学习率 (每个epoch结束后)
            if phase == 'train':
                scheduler.step()

            # 计算本轮平均Loss和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 验证阶段：保存最佳模型并检查Early Stopping
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_vehicle_model.pth')
                    print(f'Best model saved with accuracy: {best_acc:.4f}')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        # 检查是否触发早停
        if early_stop_counter >= PATIENCE:
            print(f'Early stopping triggered after {PATIENCE} epochs without improvement.')
            break

    # 输出训练总结
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()



