import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from datasets import get_data_loader
from model.model import SimpleAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

train_dataloader = get_data_loader('MNISTDataset', batch_size=64, num_workers=0, train=True, drop_last=True)
test_dataloader = get_data_loader('MNISTDataset', 16, 0, train=False, drop_last=True)

model = SimpleAutoencoder().to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters())

model.train()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        img = batch['left'].to(device)

        # Forward pass
        outputs = model(img)
        loss = criterion(outputs, img)

        # Backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        left = batch['left'].to(device)
        right = batch['right'].to(device)

        pred = model(left)
        pred = pred.view(-1, 1, 28, 14)

        # move to CPU for matplotlib
        left, pred = left.cpu(), pred.cpu()

        n = min(6, left.size(0))
        fig, axs = plt.subplots(n, 2, figsize=(4.5, n*1.5))

        for i in range(n):
            axs[i, 0].imshow(left[i, 0], cmap='gray')
            axs[i, 1].imshow(pred[i, 0], cmap='gray')
            for j in range(2):
                axs[i, j].axis('off')

        plt.tight_layout(pad=0.5)
        plt.show()
        break
