import pandas as pd
import os
import gym
import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch 
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import  DataLoader, TensorDataset
import torchvision.transforms as transforms
from run_model import NeuralNetwork
from encoder import Autoencoder
import argparse

writer = SummaryWriter()


def read_encoded_data(dagger): 
    # reads expert data and outputs state and actions
    if dagger:
        data = pd.read_pickle("./encoded_data/encoded_dagger.pkl")
        X1 = np.stack(data["state"], 0)
        y1 = np.stack(data["action"], 0)
        data2 = pd.read_pickle("./encoded_data/encoded_dataset.pkl")
        X2 = np.stack(data2["state"], 0)
        y2 = np.stack(data2["action"], 0)
        X = np.concatenate((X1,X2), 0)
        y = np.concatenate((y1,y2), 0)
    else:
        data = pd.read_pickle("./encoded_data/encoded_dataset.pkl")
        X = np.stack(data["state"], 0)
        y = np.stack(data["action"], 0)
    return X, np.array(y)

def read_data(): 
    # reads expert data and outputs state and actions
    data = pd.read_pickle("./encoded_data/encoded_dataset.pkl")
    X = np.stack(data["state"], 0)
    y = np.stack(data["action"], 0)
    return X, np.array(y)


def train(model, train_loader, criterion, optimizer, device):
    # train autoencoder 
    model.train()
    train_loss = 0
    for data in train_loader:
        inputs, label = data
        inputs = inputs.float().to(device)
        optimizer.zero_grad()
        outputs=[]
        for input in inputs:
            output = model(input)
            outputs.append(output)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, label.long())
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
            inputs, label = data
            inputs = inputs.float().to(device)
            for input in inputs:
                output = model(input)
                outputs.append(output)
            outputs = torch.stack(outputs)
            loss = criterion(outputs, label.long())
            test_loss += loss.item()
    return test_loss / len(test_loader)



if __name__ == "__main__":

     # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num_epochs", type=int, default=10)
    parser.add_argument('-b' , "--batch_size", type=int, default=32)
    parser.add_argument('-d', '--dagger', dest='dagger', action='store_true')
    parser.set_defaults(dagger=False)
    args = parser.parse_args()

    # Image transformation
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(70),
    # transforms.Grayscale(1),
    # transforms.Normalize(0.5, 0.5),
    ])

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # read data
    # X, y = read_encoded_data(args.dagger)
    X, y = read_data()
    # X_stack = []
    # for obs in X:
    #     X_stack.append(transform(obs).flatten())
    # X_stack = torch.stack(X).to(device)
    # Stack obsevations
    # stack_size = 2
    # frame_skip = 4
    # X_stack = np.array([np.concatenate((X[i], X[i-frame_skip]), 0) for i in range(frame_skip, len(X))])
    # y_stack = y[frame_skip:]
    # del X, y
    # Model parameters
    X_stack = X
    y_stack = y
    num_inputs = X_stack.shape[1]
    num_outputs = 5
    model = NeuralNetwork(num_inputs, num_outputs)
    if args.dagger:
        model.load_state_dict(torch.load('./model_weights/nn_weights.pt'))
    model.train()
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, last_epoch=- 1, verbose=False)
    # Split and load data
    # X_train, X_test, y_train, y_test = train_test_split(X_stack, y_stack, test_size=0.1, random_state=None, shuffle=None)
    training_set = TensorDataset(torch.from_numpy(X_stack), torch.from_numpy(y_stack))
    # test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # training_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # Create data loaders for the training and testing sets
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # torch.manual_seed(42)
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        # test_loss = evaluate(model, test_loader, criterion, device)
        print("Epoch:", epoch+1, " train loss: ", train_loss) #, " test loss: ", test_loss)


    # Evaluate model
    env =  gym.make('CarRacing-v2', continuous=False) #, render_mode="human")
    # load model 
    model.eval()
    # --------------------Encoder-------------------------
    # Load the saved encoder part
    encoder = Autoencoder()
    encoder.encoder = torch.load('./model_weights/good_encoder.pth')
    encoder.to(device)
    # --------------------Encoder-------------------------

    """
    # run environment
    reward = 0
    for i in range(3):
        obs = env.reset()
        reward = 0
        observations = []
        for frame in range(100000):
            obs = transform(obs).to(device).float()
            obs = encoder.encoder(obs)
            obs = obs.flatten()
            observations.append(obs)
            if frame >= frame_skip:
                obs = torch.cat((observations[0], observations[frame_skip]),0)
                act = model(obs)
                act = np.argmax(act.cpu().detach().numpy())
                del observations[0]
            else:
                act = 3
            obs, r, done, _ = env.step(act)
            env.render()
            reward += r
            if done:
                print(reward)
                break
    """
    # run environment
    reward = 0
    for i in range(3):
        obs = env.reset()
        reward = 0
        for i in range(10000):
            if i > 25:
                obs = transform(obs).float().to(device)
                obs = encoder.encoder(obs)
                obs = obs.flatten()
                act = model(obs)
                # print(np.argmax(act.cpu().detach().numpy()))
                act = np.argmax(act.cpu().detach().numpy())
            else:
                act = 3
            obs, r, done, _ = env.step(act)
            env.render()
            reward += r
            if done:
                print(reward)
                break



    # Save model
    save_dir = './model_weights'
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    if args.dagger:
        torch.save(model.state_dict(),'./model_weights/dagger_nn_weights.pt')  
    else:
        torch.save(model.state_dict(),'./model_weights/nn_weights.pt')  


