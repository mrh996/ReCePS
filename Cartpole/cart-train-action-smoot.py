import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# CartPole environment
env = gym.make('CartPole-v0')

# Hyperparameters
gamma = 0.95  # Discount factor for future rewards
epsilon = 1.0  # Exploration rate (initially fully exploratory)
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration rate
learning_rate = 0.001
memory = []

# DQN model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the model and optimizer
input_size = 4
hidden_size = 24
output_size = 2
model = DQN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def choose_action(state):
    # Epsilon-greedy action selection
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Choose a random action

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        Q_values = model(state_tensor)
        return Q_values.argmax().item()


def replay(batch_size):
    global epsilon

    batch = np.array(memory)
    states = np.vstack(batch[:, 0])
    actions = batch[:, 1]
    rewards = batch[:, 2]
    next_states = np.vstack(batch[:, 3])
    dones = batch[:, 4]

    # Convert arrays to tensors
    states_tensor = torch.FloatTensor(states)
    next_states_tensor = torch.FloatTensor(next_states)

    # Q-value prediction for the current states
    Q_values = model(states_tensor)
    # Q-value prediction for the next states
    next_Q_values = model(next_states_tensor)

    Q_target = Q_values.clone()

    for i in range(len(batch)):
        action = int(actions[i])
        if dones[i]:
            Q_target[i, action] = rewards[i]
        else:
            target = rewards[i] + gamma * torch.max(next_Q_values[i])
            Q_target[i, action] = target

    # Compute loss and backpropagate
    loss = F.mse_loss(Q_values, Q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


def train(num_episodes=1000, batch_size=32):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            # Choose an action
            action = choose_action(state)

            # Take the action and observe the new state and reward
            next_state, reward, done, _ = env.step(action)

            # Store the transition in the memory
            memory.append((state, action, reward, next_state, done))

            total_reward += reward
            state = next_state

            if done:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")
                break

            if len(memory) >= batch_size:
                replay(batch_size)


if __name__ == "__main__":
    train()