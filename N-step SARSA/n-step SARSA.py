import math

import gym
import numpy as np
import matplotlib.pyplot as plt


def nstep_sarsa(env, n=8, alpha=0.2, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_ep = []
    for episodes in range(num_ep):
        # Initialize Storage and Timer
        T = math.inf
        t = 0
        states = []
        actions = []
        rewards = []
        ep_rewards = 0
        # Store init values
        state = env.reset()
        action = choose_action(state, Q, epsilon)
        states.append(state)
        actions.append(action)
        rewards.append(0)
        # Start Loop
        while True:
            if t < T:
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                ep_rewards += reward
                if done:
                    T = t + 1
                else:
                    action = choose_action(next_state, Q, epsilon)
                    actions.append(action)
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                if tau + n < T:
                     G += gamma**n * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
            if tau == T - 1:
                break
            t += 1
            if t < T:
                state = next_state
                action = actions[t]
        rewards_per_ep.append(ep_rewards)
    return Q, rewards_per_ep


def choose_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(np.arange(env.action_space.n))
    else:
        return np.argmax(Q[state])


def plot_results(rewards, alphas, n_steps):
    plt.figure(figsize=(12, 8))
    for n in n_steps:
        avg_rewards = [np.mean(rewards[(alpha, n)][-100:]) for alpha in alphas]
        plt.plot(alphas, avg_rewards, marker='o', label=f'n={n}')
    plt.xlabel('Alpha (Learning Rate)')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.title('n-step SARSA Performance on FrozenLake-v0')
    plt.legend()
    plt.show()


env = gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
alphas = [0.2, 0.5, 0.7]
n_steps = [2, 4, 8]
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 5000  # Number of episodes for training

all_rewards = {}

for alpha in alphas:
    for n in n_steps:
        print(f'Training with α={alpha}, n={n}')
        Q, rewards = nstep_sarsa(env, n, alpha, gamma, epsilon, num_episodes)
        all_rewards[(alpha, n)] = rewards

plot_results(all_rewards, alphas, n_steps)