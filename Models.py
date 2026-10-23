import torch
from abc import ABC
import torch.nn as nn
from pathlib import Path
import torch.optim as optim


class BaseModel(nn.Module, ABC):
    def __init__(self, learning_rate=1e-3, device='cpu', seed=42):
        torch.manual_seed(seed)
        super().__init__()

        self.learning_rate = learning_rate
        self.optimizer = None
        
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.to(self.device)


    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr


    def get_device(self):
        return self.device

    def set_device(self, device):
        self.device = device
        self.to(device)


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        if Path(path).exists():
            self.load_state_dict(torch.load(path, map_location=self.device))
            return True
        else:
            return False


    def summarize(self):
        print(self)
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

        print(f"\nTotal Parameters     : {total:,}")
        print(f"Trainable Parameters : {trainable:,}")

    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError
    

class Encoder(BaseModel):
    def __init__(self, latent_size=512, learning_rate=1e-3, device="cpu", seed=42):
        super().__init__(learning_rate, device, seed)
        # --------------------------------------------------
        # Architectures
        # --------------------------------------------------
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(8 * 8 * 256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True)
        )
        self.mu = nn.Linear(2048, latent_size)
        self.logvar = nn.Linear(2048, latent_size)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

    def forward(self, x):
        h = self.feature_extractor(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def train_one_epoch(self, dataloader, criterion):
        self.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            mu, logvar = self(x)
            loss = criterion(mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
    

class Decoder(BaseModel):
    def __init__(self, latent_size, learning_rate=1e-3, device="cpu", seed=42):
        super().__init__(learning_rate, device, seed)
        # --------------------------------------------------
        # Architectures
        # --------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

    def forward(self, z):
        return self.decoder(z)

    def train_one_epoch(self, dataloader, criterion):
        self.train()
        total_loss = 0
        for z, target in dataloader:
            z = z.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self(z)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    

class VariationalAE(BaseModel):
    def __init__(self, criterion, learning_rate, light=True, device="cpu", seed=42):
        super().__init__(learning_rate, device, seed)

        self.latent_size = 512 if light else 1024
        self.criterion = criterion
        self.encoder = Encoder(
            latent_size=self.latent_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        )
        self.decoder = Decoder(
            latent_size=self.latent_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        )
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        self.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def loss_function(self, reconstruction, x, mu, logvar):
        reconstruction_loss = self.criterion(reconstruction, x,)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl

    def train_one_epoch(self, train_loader, val_loader):
        self.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self(x)
            loss = self.loss_function(reconstruction, x, mu, logvar)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(self.device)
                reconstruction, mu, logvar = self(x)
                loss = self.loss_function(reconstruction, x, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        return train_loss, val_loss
    
    def save_model(self, base_path):
        self.encoder.save_model(base_path + f'\\{self.architecture}_Encoder')
        self.decoder.save_model(base_path + f'\\{self.architecture}_Decoder')

    def load_model(self, base_path):
        main_path = base_path + f'\\{self.architecture}'
        if Path(main_path + '_Encoder').exists() and Path(main_path + '_Decoder').exists():
            self.encoder.load_model(main_path + '_Encoder')
            self.decoder.load_model(main_path + '_Decoder')
            return True
        else:
            return False
     
 
class Classifier(BaseModel):
    def __init__(self, criterion, learning_rate=1e-3, pretrained_encoder=None, device="cpu", seed=42):
        super().__init__(learning_rate, device, seed)

        self.criterion = criterion
        self.encoder = Encoder(learning_rate=learning_rate, device=device, seed=seed)
        self.pretrained_encoder = not (pretrained_encoder is None)
        if self.pretrained_encoder:
            main_path = pretrained_encoder + f'\\{self.architecture}'
            if Path(main_path + '_Encoder').exists():
                self.encoder.load_model(main_path + '_Encoder')
                self.encoder.freeze()
                print("Pre-trained model is loaded")
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 10),
        )
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        self.to(device)

    def forward(self, x):
        mu, _ = self.encoder(x)
        logits = self.classifier(mu)
        return logits

    def train_one_epoch(self, train_loader, val_loader):
        self.train()
        train_loss = 0.0
        train_total = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self(x)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_total += y.size(0)

        train_loss /= len(train_loader)

        self.eval()
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self(x)
                loss = self.criterion(logits, y)

                val_loss += loss.item()
                val_total += y.size(0)

        val_loss /= len(val_loader)

        return train_loss, val_loss
    
    def evaluate_accuracy(self, dataloader):
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self(x)
                predictions = logits.argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += y.size(0)

        return correct / total