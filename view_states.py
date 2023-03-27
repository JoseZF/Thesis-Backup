
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_data():
    # Reads encoded dataset
    path = 'encoded_data/encoded_dataset.pkl'
    encoded_data = pd.read_pickle(path)
    data = pd.read_pickle('test_data/dataset.pkl')
    enc_X = np.stack(encoded_data["state"], 0)
    X = np.stack(data['state'], 0)
    return X, enc_X 

if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num", type=int, default=5)
    args = parser.parse_args()

    X, enc_X = read_data()
    # Select 4 random pictures 
    random_imgs = np.random.choice(np.arange(len(X)), size=args.num, replace=False)
    imgs = []
    enc_imgs = []
    for i in random_imgs:
        img = X[i]
        img = np.moveaxis(img, 0, -1)
        imgs.append(img)
        img = enc_X[i]
        img = np.reshape(img, (40,40))
        enc_imgs.append(img)
    # Display the image
    plt.set_cmap('bone')
    plt.close()
    fig, axs = plt.subplots(2, args.num)
    for i in range(args.num):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(enc_imgs[i])


    plt.show()
