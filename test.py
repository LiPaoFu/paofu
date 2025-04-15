import matplotlib.pyplot as plt
import numpy as np
from SimpleUnet import model
import torch
from torch import nn
from config import CLASSES
from DataLoader import test_dataloader, test_dataset

# 初始化损失函数
criterion = nn.CrossEntropyLoss()

# 加载训练好的模型
model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 评估模式
model.eval()
test_loss = 0.0
total_samples = 0

with torch.no_grad():
    for images, masks in test_dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        test_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

# 计算平均损失
avg_test_loss = test_loss / total_samples
print(f"\nTest Loss: {avg_test_loss:.4f}")

# 重新获取一个测试样本并预测
test_img, test_mask = test_dataset[0]
test_img = test_img.unsqueeze(0).to(device)  # 添加batch维度
with torch.no_grad():
    output = model(test_img)
pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# 可视化设置
plt.figure(figsize=(15, 5))

# 显示输入图像
plt.subplot(1, 3, 1)
input_image = test_img.squeeze().permute(1, 2, 0).cpu().numpy()
input_image = (input_image * 255).astype(np.uint8)  # 反归一化
plt.imshow(input_image)
plt.title("Input Image")
plt.axis('off')

# 显示真实掩码
plt.subplot(1, 3, 2)
plt.imshow(test_mask.numpy(), cmap='jet', vmin=0, vmax=len(CLASSES) - 1)
plt.title("True Mask")
plt.colorbar(ticks=range(len(CLASSES)), orientation='horizontal')
plt.axis('off')

# 显示预测掩码
plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=len(CLASSES) - 1)
plt.title("Predicted Mask")
plt.colorbar(ticks=range(len(CLASSES)), orientation='horizontal')
plt.axis('off')

plt.tight_layout()
plt.show()