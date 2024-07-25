# Frozen Lake Policy Optimization

This project implements policy evaluation and brute-force policy optimization for the Frozen Lake environment from OpenAI Gym.

## Description

This program explores reinforcement learning techniques applied to the Frozen Lake problem. It includes:

1. Custom environment creation
2. Policy evaluation
3. Brute-force search for optimal policies

The implementation uses a 3x3 custom map for faster computation, but can be easily modified for larger environments.

## Features

- Custom Frozen Lake environment creation
- Transition probability matrix calculation
- Policy evaluation using value iteration
- Brute-force search for optimal policies
- Visualization of the environment and policies

## Requirements

- Python 3.x
- NumPy
- OpenAI Gym

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/frozen-lake-policy-optimization.git
   ```
2. Navigate to the project directory:
   ```
   cd frozen-lake-policy-optimization
   ```
3. Install the required packages:
   ```
   pip install numpy gym
   ```

## Usage

Run the main script to execute the policy evaluation and optimization:

```
python frozen_lake_policy.py
```

## Code Structure

- `trans_matrix_for_policy()`: Computes the transition probability matrix for a given policy
- `terminals()`: Identifies terminal states in the environment
- `value_policy()`: Evaluates the value function for a given policy
- `bruteforce_policies()`: Performs brute-force search to find optimal policies
- `main()`: Orchestrates the policy evaluation and optimization process

## Customization

You can modify the `custom_map3x3` variable to change the layout of the Frozen Lake environment. Alternatively, uncomment the random map generation code for larger, randomized environments.

## Output

The program outputs:
- The current Frozen Lake environment
- Value functions for predefined policies (always going left or right)
- Optimal value function
- Number of optimal policies found
- List of optimal policies

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the FrozenLake-v0 environment from OpenAI Gym and implements reinforcement learning algorithms as described in standard RL literature.
