import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

from omegaconf import OmegaConf
import pathlib

from tqdm import tqdm

from datasets import get_data_loader
from model.model import (
    SimpleAutoencoder, 
    CollaborativeAutoencoder,
    CollabConfig
)

def _prepare_output_folders(cfg):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


def train(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    _prepare_output_folders(cfg)

    train_dataloader = get_data_loader('MNISTDataset', batch_size=cfg.batch_size, num_workers=cfg.num_workers, train=True, drop_last=True)
    test_dataloader = get_data_loader('MNISTDataset', 16, 0, train=True, drop_last=True)

    model = eval(cfg.model).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters())

    model.train()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            l = batch['left'].to(device)
            r = batch['right'].to(device)
            img = [l, r]

            gt_l = batch['gt_left'].to(device)
            gt_r = batch['gt_right'].to(device)
            gt = [gt_l, gt_r]

            # Forward pass
            outputs = model(img)
            
            total_loss = 0.0
            for agent_pred, agent_gt in zip(outputs, gt[:len(outputs)]):
                loss = criterion(agent_pred, agent_gt)
                total_loss += loss

            # Backward pass and optimization
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{cfg.epochs}], Loss: {epoch_loss:.4f}')

    model.save(cfg, epoch=epoch)

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            img = [left, right][:model.collab_n_agents]
            
            gt_l = batch['gt_left'].to(device)
            gt_r = batch['gt_right'].to(device)
            gt = [gt_l, gt_r][:model.collab_n_agents]
            gt = torch.cat(gt, dim=3)

            h, w = gt.shape[2:]

            # Forward pass
            pred = model(img)
            pred = torch.cat(pred, dim=3)  # Concatenate predictions from both agents
            pred = pred.view(-1, 1, h, w)

            # move to CPU for matplotlib
            gt, pred = gt.cpu(), pred.cpu()

            n = min(6, gt.size(0))
            fig, axs = plt.subplots(n, 2, figsize=(4.5, n*1.5))

            for i in range(n):
                axs[i, 0].imshow(gt[i, 0], cmap='gray')
                axs[i, 1].imshow(pred[i, 0], cmap='gray')
                for j in range(2):
                    axs[i, j].axis('off')

            plt.tight_layout(pad=0.5)
            plt.savefig('output_comparison.png')  # Save to file
            plt.close()
            break


def run():
    cfg_path = pathlib.Path(__file__).parent / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    train(cfg)


if __name__ == "__main__":
    run()