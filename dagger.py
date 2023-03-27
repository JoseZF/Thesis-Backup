from __future__ import print_function

import argparse
import time
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import json
import pygame
import glob
import copy
import pandas as pd
from datetime import datetime
from encoder import Autoencoder
from run_model import NeuralNetwork
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

writer = SummaryWriter()

def append_df():
    # Collect all dataframes into one
    path = './dagger_data/dagger_dataset.pkl'
    # Find all the files with the .pkl extension in the current directory
    filenames = glob.glob('./dagger_data/*.pkl')

    # Create an empty DataFrame to store the results
    df_final = pd.DataFrame()

    # Iterate over the list of filenames
    for filename in filenames:
        # Load the DataFrame from the file
        df = pd.read_pickle(filename)
        
        # Append the DataFrame to the final DataFrame
        df_final = df_final.append(df, ignore_index=True)
    
    # Check if older data is available
    if os.path.isfile(path):
        # Read old data and append new data
        df = pd.read_pickle(path)
        df_final = df.append(df_final, ignore_index=True)
    
    # Delete the initial files
    for filename in filenames:
        os.remove(filename)

    # Save final df
    df_final.to_pickle(path)

def total_rewards(reward_list):
    # Save episode rewards
    path = './dagger_data/rewards/rewards.pkl'
    df_new = pd.DataFrame (reward_list, columns = ['episode_rewards'])
    # Check existing data
    if os.path.isfile(path):
        df = pd.read_pickle(path)
        df = df.append(df_new, ignore_index=True)
    else:
        df = df_new
    # Save
    df.to_pickle(path)


def register_input(a):
    global quit, restart
    # Car Controls
    if pygame.key.get_pressed()[pygame.K_LEFT]:
        a = 2
    elif pygame.key.get_pressed()[pygame.K_RIGHT]:
        a = 1
    elif pygame.key.get_pressed()[pygame.K_UP]:
        a = 3
    elif pygame.key.get_pressed()[pygame.K_DOWN]:
        a = 4
    else:
        a = 0

    # Restart and Quit
    if pygame.key.get_pressed()[pygame.K_RETURN]:
        restart = True
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        quit = True

    return a


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num_episodes", type=int, default=10)
    parser.add_argument('-b' , "--beta", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists('./dagger_data'):
        os.mkdir('./dagger_data')
    
    good_samples = {
        "state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }
    episode_samples = copy.deepcopy(good_samples)
    env = gym.make('CarRacing-v2', continuous=False, render_mode="human")
    a = 0
    episode_rewards = []
    # load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(784, 5)
    model.load_state_dict(torch.load('./model_weights/nn_weights.pt'))
    model.eval()
    model.to(device)
    # --------------------Encoder-------------------------
    # Load the saved encoder part
    loaded_encoder = torch.load('./model_weights/good_encoder.pth')
    # Use the loaded encoder
    encoder = Autoencoder().to(device)
    encoder.encoder = loaded_encoder
    # --------------------Encoder-------------------------

    all_rewards = []

    # Image transformation
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(70),
    # transforms.Grayscale(1),
    ])

    # Episode loop
    i=0
    while i < args.num_episodes:
        i += 1
        episode_samples["state"] = []
        episode_samples["action"] = []
        episode_samples["reward"] = []
        episode_samples["terminal"] = []
        obs = env.reset()
        restart = False
        quit = False
        # State loop
        while True:
            # Record observations
            episode_samples["state"].append(obs)
            # Get both actions
            # Expert actions
            expert_a = register_input(a)
            # Model actions
            obs = transform(obs).to(device).float()
            obs = encoder.encoder(obs)
            obs = obs.flatten()
            model_a = model(obs)
            model_a = np.argmax(model_a.cpu().detach().numpy())
            # Expert participation probability
            x = np.random.uniform()
            # Decide who is going to take the action
            if x >= args.beta:
                a = model_a
            else:
                a = expert_a
            # Take step
            obs, r, done, info = env.step(a)
            time.sleep(0.1)
            # Save data
            episode_samples["action"].append(expert_a)      
            episode_samples["reward"].append(r)
            episode_samples["terminal"].append(done)
            # Render
            env.render()
            # Loop end conditions
            if done or restart or quit: 
                break

        if quit:
            # If esc pressed, save previous data and exit
            break

        if not restart:
            # Delete first timesteps (get effective timesteps)
            if i == 1:
                start = 50
            else:
                start = 25
            # Create DataFrame
            df_dagger = pd.DataFrame(episode_samples)
            df_dagger.drop(df_dagger.index[:start], axis=0, inplace=True)
            
            # Saving the data
            print('... saving data')
            df_dagger.to_pickle('./dagger_data/dagger_dataset_' + str(i) + '.pkl')
            episode_rewards.append(r)

    append_df()
    total_rewards(episode_rewards)
    env.close()



