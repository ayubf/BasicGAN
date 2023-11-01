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
            nn.LeakyReLU(0.01),
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
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
# Hyperparams 
device = "cuda" if torch.cuda.is_available() else "cpu" # device to use 
lr = 3e-4 # Learning Rate
z_dim = 64 # noise dimensions
image_dim = 28 * 28 * 1 # Dimension of images, 28x28
batch_size = 32 # Batch size to divide dataset into for every epoch
num_epochs = 50 # Number of iterations

# Generator and Discriminator Init
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# Creates a batch_size number Random noise vectors of dimensions z_dim 
fixed_noise = torch.randn((batch_size, z_dim)).to(device)


# List of transforms to use on the dataset to transform it into a useable format
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Loads dataset into local dir and then into this program, downloads, and shuffles data
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimization function for the discriminator and generator weights & bias terms using Adam
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# Training Criterion, in this case Binary Cross Entropy Loss, where 0 is correct labeling and 1 is incorrect labeling
criterion = nn.BCELoss()


# For tensorboard logging
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

# Step counter, there are batch_size number of steps within an epoch
step = 0

# For every epoch in number given
for epoch in range(num_epochs):
    # Batch index and real image every item in the loader for the data set
    # The blank item is for the lablels, which the GAN doesn't use
    for batch_indx, (real, _) in enumerate(loader):
        # Turns real image input into a vector of 28x28 (2D Array) and sending it to specified device
        # Updates batch_size to shape of real image
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        
        # Training Discriminator
        noise = torch.randn(batch_size, z_dim).to(device) # Random noise generated to batch size # of random noise vectors of z_dim dimensions each
        fake = gen(noise) # Batch of fake images 
        disc_real = disc(real).view(-1) # Batch of discriminator preditions for real images
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # Compares batch of discriminator predictions by similar vector if 1s, finds loss
        disc_fake = disc(fake).view(-1) # Batch of discriminatior predictions for fake images
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Compares batch of discriminator predictions by similar vector of 0s, finds loss
        lossD = (lossD_fake + lossD_real) / 2 # Total loss
        disc.zero_grad() # Zeros out gradients
        lossD.backward(retain_graph=True) # Computes gradients of loss
        opt_disc.step() # Updates discriminator parameters using optimizer

        # Training Generator
        output = disc(fake).view(-1) # Batch of discriminator predicitions for fake images
        lossG = criterion(output, torch.ones_like(output)) # Finds loss by comparing batch of discriminator predictions for fake images by similar vector of 0s
        gen.zero_grad() # Zeros out gradients
        lossG.backward(retain_graph=True) # Computes gradient of loss
        opt_gen.step() # Updates generator parameters using optimizer

        # Only runs at the start of a batch
        if batch_indx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] \n Loss D: {lossD:.4f}, Loss G: {lossG:4f}")

            # Context manager, gradients here are not measured
            with torch.no_grad():
                # Generates fake images using fixed_noise as opposed to the fresh noise for every batch
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28) # Reshapes real images to have same dimensions as fake image

                # Create two grids from fake and real images, normalizes images 
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Add image grids to the writer objects to display on tensorboard
                writer_fake.add_image(
                    "Mnst Fake Images", img_grid_fake, global_step=step 
                )

                writer_fake.add_image(
                    "Mnst Real Images", img_grid_real, global_step=step 
                )

                step += 1 # Increment step counter