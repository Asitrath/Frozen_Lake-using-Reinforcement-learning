# n-step SARSA for FrozenLake Environment

This project implements the n-step SARSA algorithm for reinforcement learning and applies it to the FrozenLake-v0 environment from OpenAI Gym.

## Description

The n-step SARSA algorithm is a temporal difference learning method that uses n-step returns for updating the action-value function. This implementation tests the algorithm on the FrozenLake environment, specifically using an 8x8 grid.

## Features

- Implementation of n-step SARSA algorithm
- Epsilon-greedy action selection
- Evaluation of different hyperparameters (n-step sizes and learning rates)
- Visualization of results

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenAI Gym

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nstep-sarsa-frozenlake.git
   ```
2. Navigate to the project directory:
   ```
   cd nstep-sarsa-frozenlake
   ```
3. Install the required packages:
   ```
   pip install numpy matplotlib gym
   ```

## Usage

Run the main script to execute the n-step SARSA algorithm:

```
python nstep_sarsa_frozenlake.py
```

## Code Structure

- `nstep_sarsa()`: Implements the n-step SARSA algorithm
- `choose_action()`: Implements epsilon-greedy action selection
- `plot_results()`: Visualizes the results for different hyperparameters

## Hyperparameters

The script tests the following hyperparameters:
- Learning rates (α): 0.2, 0.5, 0.7
- n-step sizes: 2, 4, 8
- Fixed parameters:
  - Discount factor (γ): 0.9
  - Exploration rate (ε): 0.1
  - Number of episodes: 5000

## Output

The program outputs:
- Training progress for each combination of α and n
- A plot comparing the performance of different n-step sizes and learning rates

## Customization

You can modify the hyperparameters, number of episodes, or the environment by editing the corresponding variables in the main section of the script.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the FrozenLake-v0 environment from OpenAI Gym and implements the n-step SARSA algorithm as described in reinforcement learning literature.
