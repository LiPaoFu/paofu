import torch
from torch import nn
from HW_CV.SimpleUnet import model
from DataLoader import train_dataloader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.4f}")

# 在训练循环结束后添加
torch.save(model.state_dict(), "best_model.pth")