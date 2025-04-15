import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import os
import albumentations as A


class FloorplanDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)  # 灰度读取

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 转换为Tensor并归一化
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()

        return image, mask


# 数据增强配置（训练集专用）
train_transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
], additional_targets={'mask': 'mask'})

# 测试集仅保留必要预处理
test_transform = A.Compose([
    A.Resize(128, 128),
], additional_targets={'mask': 'mask'})

# 创建完整数据集并分割
full_dataset = FloorplanDataset("dataset/images", "dataset/masks", transform=None)  # 原始尺寸数据

# 计算分割比例
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# 分割数据集（固定随机种子保证可复现）
train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 为子集添加对应的数据增强
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# 创建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)