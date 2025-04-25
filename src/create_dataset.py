import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

def add_gaussian_noise(img, mean=0, std=0.1):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0., 1.) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))

def save_dataset():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root="data/raw", train=True, download=True, transform=transform)

    os.makedirs(r"data\raw", exist_ok=True)
    os.makedirs(r"data\noisy", exist_ok=True)

    for idx, (img_tensor, _) in enumerate(dataset):
        img = transforms.ToPILImage()(img_tensor)

        clean_path = f"data/raw/{idx}.png"
        noisy_path = f"data/noisy/{idx}.png"

        img.save(clean_path)
        noisy_img = add_gaussian_noise(img)
        noisy_img.save(noisy_path)

        if idx % 2000 == 0:
            print(f"Saved {idx} images...")

if __name__ == "__main__":
    save_dataset()
