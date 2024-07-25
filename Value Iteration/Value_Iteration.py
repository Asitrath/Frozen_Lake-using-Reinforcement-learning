import gym
import numpy as np

custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
#env = gym.make("FrozenLake-v1", desc=custom_map3x3)

# Init environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

# you can set it to deterministic with:
#env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)
# Or:
#env = gym.make("FrozenLake-v0", map_name="8x8")


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def print_policy(policy, env):
    """ This is a helper function to print a nice policy representation from the policy"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    pol = np.chararray(dims, unicode=True)
    pol[:] = ' '
    for s in range(len(policy)):
        idx = np.unravel_index(s, dims)
        pol[idx] = moves[policy[s]]
        if env.desc[idx] in [b'H', b'G']:
            pol[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in pol]))


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # env.P[state][action] gives you tuples (p, n_state, r,is_terminal),
    # which tell you the probability p that you end up in the next state n_state and receive reward r
    num_iterations = 1000

    # for every iteration
    for i in range(num_iterations):

        # update the value table, that is, we learned that on every iteration, we use the updated value
        # table (state values) from the previous iteration
        updated_v_state = np.copy(V_states)

        # now, we compute the value function (state value) by taking the maximum of Q value.

        # thus, for each state, we compute the Q values of all the actions in the state and then
        # we update the value of the state as the one which has maximum Q value as shown below:
        for state in range(n_states):
            Q_values = [sum([prob * (r + gamma * updated_v_state[s_])
                             for prob, s_, r, _ in env.P[state][action]])
                        for action in range(n_actions)]

            V_states[state] = max(Q_values)

        # after computing the value table, that is, value of all the states, we check whether the
        # difference between value table obtained in the current iteration and previous iteration is
        # less than or equal to a threshold value if it is less then we break the loop and return the
        # value table as our optimal value function as shown below:

        if np.sum(np.fabs(updated_v_state - V_states)) <= theta:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            print("Optimal value function: %s" % V_states)
            break
    
    # After value iteration algorithm, obtain policy and return it
    policy = np.zeros(n_states, dtype=int)

    # for each state
    for s in range(n_states):
        # compute the Q value of all the actions in the state
        Q_values = [sum([prob * (r + gamma * V_states[s_])
                         for prob, s_, r, _ in env.P[s][a]])
                    for a in range(n_actions)]

        # extract policy by selecting the action which has maximum Q value
        policy[s] = np.argmax(np.array(Q_values))

    return policy


def main():
    # print the environment
    print("current environment: ")
    env.reset()
    env.render()
    dims = env.desc.shape
    print()

    # run the value iteration
    policy = value_iteration()
    print("Computed policy: ")
    print(policy.reshape(dims)):
    print_policy(policy, env)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break"""


if __name__ == "__main__":
    main()
