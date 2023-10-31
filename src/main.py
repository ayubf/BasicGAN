import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Discriminator
# self.disc represents the sequence of layers for the discriminator
# Layer 1:
#   - Input size as img_dim, output size = 128
#   - Performs linear operation on input such that Output = { W  * x + b | x in input }
# Layer 2:
#   - Performs non-linear operation on input such that Output = { max(0.1x, x) | x in input } 
# Layer 3:
#   - Input size as 128, output size = 1
#   - Performs linear operation on input such that Output = ∑ n, x=1 [W * input_x + b]
# Layer 4:
#   - Performs non-linear operation such that Output =  { 1 / e ^(-x) | x in input }
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):   
        return self.disc(x)

# Generator 
# self.gen represents the sequence of layers for hte discriminator
# Layer 1:
#   - Input size as z_dim, output size as 256
#   - Performs linear operation on input such that Output = { W  * x + b | x in input }
# Layer 2:
#   - Input size as 256 (from layer above), output size as 256 
#   - Perfoms non-linear operation on input such that Output = { max(0.1x, x) | x in input } 
# Layer 3:
#   - Input size as img_dim, output size as 256
#   - Performs linear operation such that Output = ∑ n, x=1 [ W * input_x + b ]
# Layer 4:
#  - Inputs size as 256, outsputs size as 256
#  - Performs linear operation such that Output = { e^x - e^(-x) / e^x + e^(-x) | x in input } 
class Generator(nn.module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(img_dim, 256),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4