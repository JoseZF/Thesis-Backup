import os
import gym 
import json
import argparse
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from encoder import Autoencoder
import torchvision.transforms as transforms

writer = SummaryWriter()

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.PReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 32),
            nn.PReLU(),
            nn.Linear(32, num_outputs),
            nn.Softmax(dim=0),
        )
    
    def forward(self, x):
        return self.net(x.float())

# class NeuralNetwork(nn.Module):
#     def __init__(self, num_inputs: int, num_outputs: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(num_inputs, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16),
#             nn.ReLU(),
#             nn.Linear(16, num_outputs),
#             nn.Softmax(dim=0),
#         )
    
#     def forward(self, x):
#         return self.net(x.float())

def run_env(model, env, device, max_steps = 10000):
    # Create game loop
    reward = 0
    obs = env.reset()
    # --------------------Encoder-------------------------
    # Load encoder
    encoder = Autoencoder().to(device)
    encoder.encoder = torch.load('./model_weights/good_encoder.pth')
    # --------------------Encoder-------------------------

    # Image transformation
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(60),
    # transforms.Grayscale([1]),
    ])

    while True:
        obs = transform(obs).to(device).float()
        obs = encoder.encoder(obs)
        obs = obs.flatten()
        act = model(obs)
        act = np.argmax(act.cpu().detach().numpy())
        obs, r, done, _ = env.step(act)
        env.render()
        reward += r
        if done:
            break


    return reward

def save_test_results(episode_rewards, dir):
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    if not os.path.exists(dir):
        os.mkdir(dir)
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{dir}results_model-{time_stamp}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n' , "--num_episodes", type=int, default=10)
    parser.add_argument('-d', '--dagger', dest='dagger', action='store_true')
    parser.set_defaults(dagger=False)
    args = parser.parse_args()

    # create environment
    env =  gym.make('CarRacing-v2', continuous=False, render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model 
    model = NeuralNetwork(22*22, 5)
    if args.dagger:
        model.load_state_dict(torch.load('./model_weights/dagger_nn_weights.pt'))
    else:
        model.load_state_dict(torch.load('./model_weights/good_nn_weights.pt'))
    model.eval()
    model.to(device)

    # run environment
    all_rewards = []
    for i in range(args.num_episodes):
        reward = run_env(model, env, device)
        all_rewards.append(reward)
        print(f'Episode {i+1} reward:{reward:.2f}')
    env.close()

    save_test_results(all_rewards, './test_results/')

