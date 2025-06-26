import torch.nn as nn

class DepthEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(DepthEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: [batch, 1, 480, 640]
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # [32, 240, 320]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # [64, 120, 160]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [128, 60, 80]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [256, 30, 40]
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 30 * 40, latent_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc(x)

class DepthDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(DepthDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 30 * 40)
        self.unflatten = nn.Unflatten(1, (256, 30, 40))
        self.deconv_layers = nn.Sequential(
            # Input: [256, 30, 40]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # [128, 60, 80]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # [64, 120, 160]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, 
                              padding=2, output_padding=1),  # [32, 240, 320]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, 
                              padding=2, output_padding=1),  # [1, 480, 640]
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.fc(z)
        z = self.unflatten(z)
        return self.deconv_layers(z)

class DepthAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(DepthAutoencoder, self).__init__()
        self.encoder = DepthEncoder(latent_dim)
        self.decoder = DepthDecoder(latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)