import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import matplotlib.pyplot as plt
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from model import ConvolutionalAutoEncoder, VGGFeatureExtractor
from PIL import Image
from tqdm import tqdm
import random
import os
import mlflow
import mlflow.pytorch

class NoisyCIFAR10(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_paths = sorted(os.listdir(noisy_dir))
        self.clean_paths = sorted(os.listdir(clean_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy = Image.open(os.path.join(self.noisy_dir, self.noisy_paths[idx]))
        clean = Image.open(os.path.join(self.clean_dir, self.clean_paths[idx]))

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)

        return noisy, clean

dataset_CIFAR10 = NoisyCIFAR10(r'data\noisy',r'data\raw',transform=transforms.ToTensor())
print("Created Dataset")
noisyimg,clean = dataset_CIFAR10[100]
plt.imshow(noisyimg.permute(1,2,0))
plt.show()
plt.imshow(clean.permute(1,2,0))
plt.show()

train_size = 40000
validation_size = 10000
train_dataset, val_dataset = random_split(dataset_CIFAR10, [train_size, validation_size])

vgg_extractor = VGGFeatureExtractor()
print("Created VGG19 Feature Extractor")
def perceptual_loss(output, target):
    output_feat = vgg_extractor(output)
    target_feat = vgg_extractor(target)

    loss = 0.0
    for layer in output_feat:
        loss += torch.nn.functional.mse_loss(output_feat[layer], target_feat[layer])
    return loss

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print("Created Dataloader")

ConvAE = ConvolutionalAutoEncoder()
criterionConvAE = nn.MSELoss()
optimiserConvAE = optim.Adam(ConvAE.parameters(),lr = 0.001)
print("Initialised Model")

def batch_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1) 
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr

def ssim(img1,img2):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_value = ssim_metric(img1, img2)
    return ssim_value.item()

mlflow.set_experiment("cifar10_image_denoising_training") 

def train_model_conv(model,dataloader,optimiser,criterion,num_epochs,alpha):
    model.train()
    std_dev = 0.05
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": num_epochs,
            "batch_size": dataloader.batch_size,
            "learning_rate": optimiser.param_groups[0]['lr'],
            "alpha": alpha,
            "noise_std_dev": std_dev
        })
        for epochs in range(1,num_epochs + 1):
            i = 0
            train_loss = 0
            vgg_train_loss = 0
            mse_train_loss = 0
            PSNR_list = []
            SSIM_list = []
            loop = tqdm(dataloader, desc=f"Epoch [{epochs}/{num_epochs}]", leave=False)
            for n_batch,c_batch in loop:
                noisy_batch = n_batch + (std_dev*torch.randn(n_batch.shape))
                noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)
                reconstructed_X = model(noisy_batch)
                optimiser.zero_grad()
            
                mse_loss = criterion(reconstructed_X,c_batch)
                mse_train_loss += mse_loss.item()
                if i%200 == 0:
                    reconstructed_X_resized = F.interpolate(reconstructed_X, size=(224, 224), mode='bilinear', align_corners=False)
                    batch_resized = F.interpolate(c_batch, size=(224, 224), mode='bilinear', align_corners=False)
                    vgg_loss = perceptual_loss(reconstructed_X_resized,batch_resized)
                    vgg_train_loss += vgg_loss.item()
                    total_loss = alpha*mse_loss + (1 - alpha)*vgg_loss
                else:
                    total_loss = mse_loss
                total_loss.backward()
                optimiser.step()
                train_loss += total_loss.item()
                i += 1
        
            model.eval()
            loop_2 = tqdm(val_dataset, desc=f"Evaluating PSNR and SSIM", leave=False)
            for noisy_img,clean_img in loop_2:
                noisy_img = noisy_img.unsqueeze(dim = 0)
                with torch.no_grad():
                    restored_img = model(noisy_img)
                psnr = batch_psnr(restored_img,clean_img.unsqueeze(dim = 0))
                SSIM = ssim(restored_img,clean_img.unsqueeze(dim = 0))
                PSNR_list.append(psnr.item())
                SSIM_list.append(SSIM)
        
            avg_psnr = sum(PSNR_list)/len(PSNR_list)
            avg_ssim = sum(SSIM_list)/len(SSIM_list)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "mse_loss": mse_train_loss,
                "vgg_loss": vgg_train_loss,
                "psnr": avg_psnr,
                "ssim": avg_ssim
                }, step=epochs)
            
            print(f"Epoch [{epochs}/{num_epochs}] || training_loss: {train_loss:.4f} || mse_loss: {mse_train_loss:.4f} || vgg_loss: {vgg_train_loss:.4f} || PSNR: {avg_psnr:.4f} || SSIM: {avg_ssim:.4f}")
            torch.save(model.state_dict(), "models/denoising_model.pth")
            model.train()

epochs = 50
print(f"-----Training model for {epochs} epochs-----")
train_model_conv(ConvAE,train_dataloader,optimiserConvAE,criterionConvAE,epochs,0.8)