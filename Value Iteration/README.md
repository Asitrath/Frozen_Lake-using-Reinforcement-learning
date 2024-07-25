# Frozen Lake Value Iteration

This project implements the Value Iteration algorithm for solving the Frozen Lake environment from OpenAI Gym.

## Description

This program demonstrates the use of Value Iteration, a dynamic programming method, to find the optimal policy for the Frozen Lake problem. The Frozen Lake environment is a grid world where an agent must navigate from a start state to a goal state while avoiding holes.

## Features

- Custom Frozen Lake environment creation
- Implementation of Value Iteration algorithm
- Policy extraction from computed value function
- Visualization of the computed policy

## Requirements

- Python 3.x
- NumPy
- OpenAI Gym

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/frozen-lake-value-iteration.git
   ```
2. Navigate to the project directory:
   ```
   cd frozen-lake-value-iteration
   ```
3. Install the required packages:
   ```
   pip install numpy gym
   ```

## Usage

Run the main script to execute the Value Iteration algorithm:

```
python frozen_lake_value_iteration.py
```

## Code Structure

- `print_policy()`: Helper function to print a visual representation of the policy
- `value_iteration()`: Implements the Value Iteration algorithm
- `main()`: Orchestrates the environment setup, algorithm execution, and result visualization

## Customization

You can modify the Frozen Lake environment by:
- Uncommenting the custom map creation (`custom_map3x3`)
- Using a randomly generated map
- Changing the size of the environment (e.g., to 8x8)
- Making the environment deterministic by setting `is_slippery=False`

## Output

The program outputs:
- The current Frozen Lake environment
- The computed optimal policy
- A visual representation of the policy

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project uses the FrozenLake-v1 environment from OpenAI Gym and implements the Value Iteration algorithm as described in reinforcement learning literature.
