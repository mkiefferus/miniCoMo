from datasets import get_data_loader

dataloader = get_data_loader('MNISTDataset', 4, 2, True, True)

# matplotlib inline
import random
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

# Get first batch
images, labels = next(iter(dataloader))

# Remove channel dim and convert to numpy
images_2_show = [img[0].numpy() for img in images]
titles_2_show = [f"label = {label.item()}" for label in labels]

show_images(images_2_show, titles_2_show)