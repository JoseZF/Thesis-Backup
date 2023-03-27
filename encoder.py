import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    # Class to extract key features from image
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Convolution of original image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Deconvolution of concoluted image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=1, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=8, stride=2, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # forward pass through autoencoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    # train autoencoder 
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = []
        for i in data:
            output = model(i)
            outputs.append(output)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    # Evaluate how well the image is reconstructed
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = []
            for i in data:
                output = model(i)
                outputs.append(output)
            outputs = torch.stack(outputs)
            loss = criterion(outputs, data)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def visualize(model, test_loader, criterion, device):
    # Evaluate how well the image is reconstructed
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = []
            for i in data:
                output = model(i)
                outputs.append(output)
            outputs = torch.stack(outputs)
            encoder_output = model.encoder(data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
            break
    return test_loss / len(test_loader), outputs[0], data[0], encoder_output[0] #* (1/255)

def read_data(data_dir="./test_data"):
    # Get state data
    data_file = os.path.join(data_dir, 'dataset.pkl')
    data = pd.read_pickle(data_file)
    # Preprocess states
    X = np.stack(data["state"], 0)
    return X

if __name__ == "__main__":

    # Arguments to pass
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num_epochs", type=int, default=100)
    args = parser.parse_args()

    # Image Transforms
    transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(70),
    # transforms.Grayscale(1),
    # transforms.Normalize(0.5, 0.5), 
])
    
    # NN setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Load the saved encoder part
    # loaded_encoder = torch.load('./model_weights/encoder.pth')
    # Use the loaded encoder
    model = Autoencoder().to(device)
    model = torch.load('./model_weights/autoencoder.pth').to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Data setup
    X = read_data()

    # Split and load data
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=None, shuffle=True)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=64, shuffle=None)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=64, shuffle=None)

    # Training Loop
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print("Epoch: ", epoch, "train loss: ", train_loss)
        test_loss = evaluate(model, test_loader, criterion, device)
        print("Epoch: ", epoch, "test loss: ", test_loss)
    test_loss, net_img, og_img, enc_img = visualize(model, test_loader, criterion, device)

    # Transform Images to RGB
    og_img = og_img.cpu().numpy()
    net_img = net_img.cpu().numpy()
    enc_img = enc_img.cpu().numpy()
    # Convert the image from (C, H, W) to (H, W, C) format
    og_img = np.transpose(og_img, (1, 2, 0))
    net_img = np.transpose(net_img, (1, 2, 0))
    enc_img = np.transpose(enc_img, (1, 2, 0))
    # Display the image
    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(og_img)
    axs[2].imshow(net_img)
    axs[1].imshow(enc_img)
    plt.show()

    # Save only the encoder part
    torch.save(model, 'model_weights/autoencoder.pth')
    torch.save(model.encoder, 'model_weights/encoder.pth')