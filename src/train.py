import torch
import torch.nn as nn
from torch.nn import MSELoss
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

def test_one_epoch(model, criterion, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            img = [batch['left'].to(device), batch['right'].to(device)][:model.collab_n_agents]
            gt = [batch['gt_left'].to(device), batch['gt_right'].to(device)][:model.collab_n_agents]

            outputs = model(img)
            loss = sum(criterion(pred, truth) for pred, truth in zip(outputs, gt[:len(outputs)]))
            total_loss += loss.item()

            for pred, truth in zip(outputs, gt[:len(outputs)]):
                pred_bin = (pred > 0.5).int()
                truth_bin = (truth > 0.5).int()
                correct += (pred_bin == truth_bin).sum().item()
                total += truth_bin.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Test Epoch: [{epoch}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy


def train_one_epoch(model, dataloader, criterion, optimiser, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}"):
        img = [batch['left'].to(device), batch['right'].to(device)]
        gt = [batch['gt_left'].to(device), batch['gt_right'].to(device)]

        outputs = model(img)
        total_loss = sum(criterion(pred, truth) for pred, truth in zip(outputs, gt[:len(outputs)]))

        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.4f}')


def train(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    _prepare_output_folders(cfg)

    # Load datasets
    train_dataloader = get_data_loader(
        cfg.train_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        train=True, 
        drop_last=True
    )
    test_dataloader = get_data_loader(
        cfg.test_dataset, 
        batch_size=16,
        num_workers=0, 
        train=True, 
        drop_last=True
    )

    # ===== Load Model =====

    start_epoch = 0
    if cfg.from_pretrained:
        model, _, last_epoch = eval(cfg.model).from_pretrained(cfg.from_pretrained)
        start_epoch = last_epoch + 1
    else:
        model = eval(cfg.model).to(device)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_criterion = eval(cfg.train_criterion)
    test_criterion = eval(cfg.test_criterion)
    optimiser = optim.Adam(model.parameters())

    # ===== Train =====

    try:
        for epoch in range(start_epoch, cfg.epochs):
            # Train for one epoch
            train_one_epoch(model, train_dataloader, train_criterion, optimiser, device, epoch, cfg.epochs)

            # Test after each epoch
            test_loss, test_acc = test_one_epoch(model, test_criterion, test_dataloader, device, epoch)

            model.check_best(test_loss, epoch)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Saving best model and exiting...")

    finally:
        # Save final model
        model.save(cfg, epoch, "model_best", save_best=True)

    # ===== Visualisation =====

    model.load_best()
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