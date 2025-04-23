import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from diffusers import AutoencoderKL


def generate_latents(
    src_folder: str,
    dst_folder: str,
    img_size: int = 256,
    batch_sz: int = 32,
    use_amp: bool = True,
    skip_existing: bool = False,
    device: torch.device = None,
):
    """
    Encodes all images in `src_folder` into latents using a pretrained VAE
    and saves each latent as a .npy file in `dst_folder`.
    """
    # Resolve paths and create output directory
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained VAE to device
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = vae.to(device)
    vae.eval()

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True)
    ])

    # Create dataset & loader
    dataset = ImageFolder(str(src_path), transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    file_counter = 0
    # Loop over batches
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Encoding batches"):
            batch_imgs = batch_imgs.to(device)
            # Optionally use mixed precision
            with (torch.cuda.amp.autocast() if use_amp else torch.no_grad()):
                # Encode to latent space and scale
                posteriors = vae.encode(batch_imgs).latent_dist
                latents = posteriors.sample() * 0.18215

            # Move to CPU and save each example
            latents_cpu = latents.cpu().numpy()
            for latent in latents_cpu:
                out_file = dst_path / f"{file_counter:06d}.npy"
                if skip_existing and out_file.exists():
                    file_counter += 1
                    continue
                np.save(out_file, latent)
                file_counter += 1

    print(f"Saved {file_counter} latent files to {dst_path}")


if __name__ == "__main__":
    # Example usage
    generate_latents(
        src_folder="./dataset/celeb_hq_256",
        dst_folder="./dataset/latent",
        img_size=256,
        batch_sz=32,
        use_amp=True,
        skip_existing=True
    )