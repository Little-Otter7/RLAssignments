import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a Buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

def run_training(algorithm, env_name, episodes):
    rewards = []
    for _ in range(episodes):
        reward = algorithm(env_name, 1)  # Train for 1 episode at a time
        rewards.append(reward)
    return rewards

# Plotting function
def plot_results(dqn_rewards, double_dqn_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(dqn_rewards, axis=0), label='DQN')
    plt.plot(np.mean(double_dqn_rewards, axis=0), label='Double DQN')
    plt.title('DQN vs Double DQN Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    dqn_rewards = run_training(train_dqn, 'MountainCar-v0', 100)
    double_dqn_rewards = run_training(train_double_dqn, 'MountainCar-v0', 100)
    plot_results(dqn_rewards, double_dqn_rewards)
