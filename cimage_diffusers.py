from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)


# Move the pipeline to the GPU
pipeline = pipeline.to("cuda")

# Generate an image with the pipeline
result = pipeline("An image of a squirrel in Picasso style")
image = result.images[0]

# # Show the image
# image.show()

# Save the image using matplotlib
plt.imshow(image)
plt.axis("off")  # Hide axes for clarity
plt.savefig('test_diffusion.png', bbox_inches='tight')
plt.close()

# So Diffusion works

import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.label_embed = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim)
        )

    def forward(self, labels):
        return self.label_embed(labels)
    
from diffusers import DDPMScheduler

scheduler = DDPMScheduler(num_train_timesteps=1000)

from diffusers import UNet2DConditionModel

unet_config = {
    "sample_size": 32,
    "in_channels": 4,
    "out_channels": 4,
    "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
    "block_out_channels": (128, 256),
    "cross_attention_dim": 64,
    "layers_per_block": 2,
    "norm_num_groups": 32,
}

unet = UNet2DConditionModel(**unet_config)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((32, 32))
])

class FloatLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label = torch.tensor(label, dtype=torch.float32) / 9.0  # Normalize label to [0, 1]
        return img, label

mnist_train = FloatLabelDataset(datasets.MNIST(root='./data', train=True, download=True, transform=transform))
train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet.to(device)
label_embed = LabelEmbedding(input_dim=1, embed_dim=unet_config['cross_attention_dim']).to(device)

optimizer = torch.optim.AdamW(list(unet.parameters()) + list(label_embed.parameters()), lr=1e-4)

for epoch in range(5):  # Example with 5 epochs
    unet.train()
    label_embed.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        # Generate noise and timesteps
        noise = torch.randn((images.size(0), 4, 32, 32), device=device)  # Latent dim = 4, image size = 32
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.size(0),), device=device).long()

        # Add noise to the images
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Embed labels into encoder_hidden_states
        encoder_hidden_states = label_embed(labels.unsqueeze(-1)).unsqueeze(1)  # Add sequence dimension

        # Predict noise
        noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # Compute loss
        loss = nn.MSELoss()(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/5 | Loss: {loss.item()}")

# Save the trained UNet and LabelEmbedding models
torch.save(unet.state_dict(), "unet_model.pth")
torch.save(label_embed.state_dict(), "label_embed_model.pth")

print("Models saved successfully!")

