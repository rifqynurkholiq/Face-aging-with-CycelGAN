import numpy as np
import torch
from torchvision.models import inception_v3
from gan_module import AgingGAN

def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    eps = 1e-6
    sqrt_cov_real = np.linalg.cholesky(sigma_real + eps)
    diff_mu = mu_real - mu_generated
    
    fid = np.sum(diff_mu**2) + np.trace(sigma_real + sigma_generated - 2 * np.sqrt(sigma_real @ sigma_generated))
    
    return fid

def compute_fid(generator, real_loader):
    inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    inception_model.eval()
    
    # Extract features from real images
    real_features = []
    for real_batch, _ in real_loader:
        real_features.append(inception_model(real_batch).detach().numpy())
    real_features = np.concatenate(real_features, axis=0)
    
    # Generate fake images and extract features
    fake_features = []
    generator.eval()
    with torch.no_grad():
        for real_batch, _ in real_loader:
            fake_batch = generator(real_batch.to(generator.device)).cpu()
            fake_features.append(inception_model(fake_batch).detach().numpy())
    fake_features = np.concatenate(fake_features, axis=0)
    
    fid_score = calculate_fid(real_features, fake_features)
    return fid_score

# model = AgingGAN()

# # Ganti genA2B dan train_dataloader() dengan instance dan fungsi yang sesuai jika Anda memanggil ini dari dalam kelas
# fid_score = compute_fid(model.genA2B, model.train_dataloader())
# print('FID:', fid_score)
