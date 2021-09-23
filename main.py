import argparse
from collections import namedtuple
import math
import random
import os
import numpy as np
import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import ale_py

# custome modules
from wrappers import *
from memory import ReplayMemory
from models import *


# argparser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-bs",
    "--batch_size",
    nargs="?",
    type=int,
    default=32,
    help="Number of batch size.",
)
parser.add_argument(
    "-epi",
    "--n_episodes",
    nargs="?",
    type=int,
    default=500,
    help="Number of training episodes.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    nargs="?",
    type=float,
    default=1e-4,
    help="learning rate",
)
args = parser.parse_args()


Transition = namedtuple("Transion", ("state", "action", "next_state", "reward"))


def select_action(state):
    """
    epsilon-greedy policy
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    ).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # # for check
    # print("non_final_mask:", non_final_mask.shape)
    # print()
    # print("state_batch:", state_batch.shape)
    # print("action_batch:", action_batch.shape)
    # print("reward_batch:", reward_batch.shape)
    # print("non_final_next_states:", non_final_next_states.shape)
    # print()
    # print("state_action_values:", state_action_values.shape)
    # print("expected_state_action_values:", expected_state_action_values.shape)
    # quit()

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        while True:
            action = select_action(state)
            if render:
                env.render()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to("cpu"), next_state, reward.to("cpu"))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if episode % 20 == 0:
            print(
                "Total steps: {} \t Episode: {}/{} \t Total reward: {}".format(
                    steps_done, episode, n_episodes, total_reward
                )
            )
    env.close()
    model_folder = check_path("model")
    save_folder = os.path.join(model_folder, fname)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
        print("Create path:", save_folder)
    torch.save(policy_net.state_dict(), os.path.join(save_folder, "policy_net.pt"))


def test(env, n_episodes, policy, render=True):
    save_folder = check_path("videos")
    env = gym.wrappers.Monitor(env, os.path.join(save_folder, fname), force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        while True:
            action = policy(state.to(device)).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print(
                    "Finished Episode {} with reward {}".format(episode, total_reward)
                )
                break

    env.close()


def check_path(fname):
    """
    Check whether folder exist or not.
    """
    pwd = os.getcwd()
    save_folder = os.path.join(pwd, fname)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
        print("Create path:", save_folder)
    return save_folder


if __name__ == "__main__":

    # create environment
    # env_name = "PongNoFrameskip-v4"
    env_name = r"PongDeterministic-v4"
    env = gym.make(env_name)
    env = make_env(env)
    print("num of actions:", env.action_space.n)
    print("action meaning:", env.unwrapped.get_action_meanings())
    n_action = env.action_space.n

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = args.batch_size
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = args.learning_rate
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    n_episodes = args.n_episodes

    # folder name
    fname = env_name + "_epo" + str(n_episodes)

    # create networks
    policy_net = DQNbn(n_actions=n_action).to(device)
    target_net = DQNbn(n_actions=n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # setup step counter
    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    train(env, n_episodes)

    # load model
    model_path = os.path.join("model", fname, "policy_net.pt")
    policy_net = DQNbn(n_actions=n_action).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    print("load model:", model_path)

    # test model
    test(env, 1, policy_net, render=False)
