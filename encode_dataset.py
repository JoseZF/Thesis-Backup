from encoder import Autoencoder
import numpy as np 
import pandas as pd
import torch
import os
import argparse
import torchvision.transforms as transforms

def read_data(data_dir="./original_data"):
    # Get state data
    data = pd.read_pickle(data_dir)
    return data.copy(deep=True)


if __name__ == "__main__":
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dagger', dest='dagger', action='store_true')
    parser.set_defaults(dagger=False)
    args = parser.parse_args()

    # Image transformation
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(70),
    # transforms.Grayscale([1]),
    ])

    # --------------------Encoder-------------------------
    # Set up GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Load the saved encoder part
    loaded_encoder = torch.load('./model_weights/good_encoder.pth')
    # Use the loaded encoder
    encoder = Autoencoder().to(device)
    encoder.encoder = loaded_encoder.to(device)
    # --------------------Encoder-------------------------

    # read data
    if args.dagger:
        data_dir = "dagger_data/dagger_dataset.pkl"
    else:
        data_dir = "test_data/dataset.pkl"
    data = read_data(data_dir)
    # Preprocess states

    X = data["state"]
    X = np.array(np.stack(X, 0))

    
    # Encode Images
    imgs = []
    for i, img in enumerate(X):
        # img = transform(img).float().to(device)
        img = torch.from_numpy(img).float().to(device)
        imgs.append(encoder.encoder(img).detach().cpu().numpy().flatten())
    
    # Save in DataFrame format (484 datapoints)
    new_df = data
    new_df["state"] = imgs
    if args.dagger:
        new_df.to_pickle("./encoded_data/encoded_dagger.pkl")
    else:
        new_df.to_pickle("./encoded_data/encoded_dataset.pkl")



    
    
        
    