# gate.py

import torch
import torch.nn as nn
import torch.optim as optim


class GateAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_gate_autoencoder(model, dataloader, device, epochs=5, lr=1e-3):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0

        for images, _ in dataloader:
            images = images.to(device)
            x = images.view(images.size(0), -1)

            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        print(f"[Gate AE] Epoch {epoch+1}/{epochs} Loss: {running_loss / max(count, 1):.6f}")

    return model


@torch.no_grad()
def compute_avg_reconstruction_error(model, dataloader, device, max_batches=10):
    model.eval()
    model.to(device)

    criterion = nn.MSELoss(reduction="none")

    total_error = 0.0
    total_samples = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        x = images.view(images.size(0), -1)

        recon = model(x)
        per_sample = criterion(recon, x).mean(dim=1)

        total_error += per_sample.sum().item()
        total_samples += x.size(0)

    if total_samples == 0:
        return float("inf")

    return total_error / total_samples
