# FrozenLake SARSA and Q-Learning Algorithms

This project implements and compares two reinforcement learning algorithms, SARSA and Q-learning, on the FrozenLake environment from OpenAI's Gym library. The goal is to navigate a frozen lake and reach a goal while avoiding holes using the two different learning algorithms.

## Environment

The FrozenLake environment consists of a grid where the agent starts at the beginning and needs to reach the goal. Some cells contain holes, and if the agent steps into a hole, it falls and the episode ends.

## Installation

To run this project, you need Python and several libraries installed. The primary library is OpenAI's Gym. You can install all the dependencies using the following command:

```bash
pip install gym numpy matplotlib
```

## Usage

The main code is implemented in a single Python file. You can run it directly to see the results of the SARSA and Q-learning algorithms.

```bash
python frozen_lake_rl.py
```

### Code Structure

- **`print_policy(Q, env)`**: Helper function to print the policy derived from the Q function.
- **`plot_V(Q, env)`**: Helper function to plot the state values from the Q function.
- **`plot_Q(Q, env)`**: Helper function to plot the Q function.
- **`sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=10000)`**: Implements the SARSA algorithm.
- **`qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=10000)`**: Implements the Q-learning algorithm.
- **`epsilon_greedy(Q, epsilon, n_actions, s)`**: Helper function to select an action using the epsilon-greedy policy.
- **`plot_average_episode_length(episode_lengths, window=100)`**: Plots the average episode length over time.
- **`plot_average_reward(reward_array, window=100)`**: Plots the average reward over time.

### Parameters

- **`alpha`**: Learning rate.
- **`gamma`**: Discount factor.
- **`epsilon`**: Exploration rate.
- **`num_ep`**: Number of episodes.


### Results

The results include the policy and value functions derived from both SARSA and Q-learning.

**Sarsa-** Plot of the state-value function & action-value function

![image](https://github.com/user-attachments/assets/a6daacb4-0d48-42d3-8a7e-7cb5e7c8c414)

**Q-Learning-** Plot of the state-value function & action-value function

![image](https://github.com/user-attachments/assets/2e19e054-f167-4c7b-a204-499ee9a4e06f)


## License

This project is licensed under the MIT License.

## Acknowledgments

- [OpenAI Gym](https://gym.openai.com/)

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.
