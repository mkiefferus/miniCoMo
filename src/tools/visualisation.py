import torch
import matplotlib.pyplot as plt

def show_images(images, title_texts):
    cols = 5
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(10, 6))  # Smaller canvas
    index = 1    
    for img, title in zip(images, title_texts):        
        plt.subplot(rows, cols, index)        
        plt.imshow(img, cmap=plt.cm.gray)
        if title:
            plt.title(title, fontsize=8)  # Smaller text
        plt.axis('off')  # Optional: hide axes
        index += 1
    plt.tight_layout()
    plt.show()


def vis_model_output(model, dataloader, output="example_output.png", device='cpu'):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
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
            plt.savefig(output)  # Save to file
            plt.close()
            break