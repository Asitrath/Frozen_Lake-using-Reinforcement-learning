import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper',
               extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6,
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0, vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            plt.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                     horizontalalignment='center',
                     verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            plt.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # implement the sarsa algorithm
    episode_lengths = np.zeros(num_ep)  # Track episode lengths
    # Q(St,At) ← Q(St,At) + α[Rt+1 + γQ(St+1,At+1) − Q(St,At)]
    reward_array = np.zeros(num_ep)
    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        s = env.reset() #state
        done = False
        total_reward = 0
        episode_length = 0
        # epsilon  greedy
        a = epsilon_greedy(Q, epsilon, env.action_space.n, s)
        while not done:
            # observe R, St+1
            s_, r, done, _ = env.step(a) # next_sate, reward, done

            total_reward += r
            episode_length += 1

            a_ = epsilon_greedy(Q, epsilon, env.action_space.n, s_) #chose At+1 from St+1
            #update Q values
            Q[s,a] += alpha*(r+(gamma*Q[s_,a_])- Q[s,a])
            s,a = s_, a_
        reward_array[i] = total_reward
        episode_lengths[i] = episode_length
    return Q,reward_array, episode_lengths


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # implement the qlearning algorithm
    episode_lengths = np.zeros(num_ep)  # Track episode lengths
    # Q(St,At) ← Q(St,At) + α[Rt+1 + γ*max[Q(St+1,A)] − Q(St,At)]
    reward_array = np.zeros(num_ep)
    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        s = env.reset() #state
        done = False
        total_reward = 0
        episode_length = 0
        while not done:
            # epsilon  greedy
            a = epsilon_greedy(Q, epsilon, env.action_space.n, s)
            # observe R, St+1
            s_, r, done, _ = env.step(a) # next_sate, reward, done

            total_reward += r
            episode_length += 1

            #update Q values
            Q[s,a] = Q[s,a] + alpha*(r+(gamma*np.max(Q[s_:]))- Q[s,a])
            s = s_
        reward_array[i] = total_reward
        episode_lengths[i] = episode_length
    return Q,reward_array, episode_lengths

def epsilon_greedy(Q, epsilon, n_actions, s):
    """
    Q: Q Table
    epsilon: exploration parameter
    n_actions: number of actions
    s: state
    """
    # selects a random action with probability epsilon
    if np.random.random() <= epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[s, :])

def plot_average_episode_length(episode_lengths, window=100):
    avg_lengths = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    plt.plot(avg_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length')
    plt.title('Average Episode Length Over Time')
    plt.show()

def plot_average_reward(reward_array, window=100):
    avg_rewards = np.convolve(reward_array, np.ones(window) / window, mode='valid')
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.show()

# env = gym.make('FrozenLake-v1')
env=gym.make('FrozenLake-v1', is_slippery=False)
# env=gym.make('FrozenLake-v1', map_name="8x8")

print("current environment: ")
env.reset()
env.render()
print()

print("Running sarsa...")
Q, reward_array, episode_lengths = sarsa(env)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()
plot_average_episode_length(episode_lengths) # Plot the average episode length
#plot_average_reward(reward_array)

print("\nRunning qlearning")
Q, reward_array, episode_lengths = qlearning(env)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()
plot_average_episode_length(episode_lengths) # Plot the average episode length
#plot_average_reward(reward_array)
