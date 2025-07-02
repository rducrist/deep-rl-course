import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDLVM(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder: maps 1D latent z to 2 pixel probabilities
        self.fc = nn.Linear(1, 2)

    def forward(self, z):
        # Output Bernoulli probabilities with sigmoid
        logits = self.fc(z)  # shape [batch_size, 2]
        probs = torch.sigmoid(logits)
        return probs

# Log-likelihood of binary data under Bernoulli model
def log_bernoulli(x, probs):
    return torch.sum(x * torch.log(probs + 1e-9) + (1 - x) * torch.log(1 - probs + 1e-9), dim=1)

# Toy data: batch of 3 samples, 2 binary pixels each
x_data = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

model = SimpleDLVM()

# Sample latent variable z from prior N(0,1), batch size = 3
z = torch.randn(3, 1)

# Decode latent z to pixel probabilities
probs = model(z)

# Compute log-likelihood of each sample
log_px_z = log_bernoulli(x_data, probs)

print("Latent variables z:\n", z)
print("Decoded Bernoulli probs:\n", probs)
print("Log likelihood log p(x|z):\n", log_px_z)

# Generate new samples by sampling from prior and decoding
with torch.no_grad():
    z_new = torch.randn(5,1)
    probs_new = model(z_new)
    # Sample pixels from Bernoulli probabilities
    samples = torch.bernoulli(probs_new)
    print("\nGenerated samples from prior:")
    print(samples)
