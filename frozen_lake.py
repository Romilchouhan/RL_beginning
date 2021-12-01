import gym
import numpy as np
env = gym.make('FrozenLake-v1')

# no. of states in environment is 16
print(env.observation_space.n)

# no. of actions in the environment is 4
print(env.action_space.n)

###########################################
##       Value interation function       ##
###########################################

def value_iteration(env, gamma = 1.0):
    # first initialize random value function which is 0 for all states
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000

    # then upon starting each iteration, copy the value_table to updated_value_table
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)

        # Now calculate Q value for each state
        # We create a list called Q_value
        # then for each action in the state, we create a list called next_state_rewards
        # we sum the next_state_rewards and append it to our Q_value

        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_state_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_state_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
                Q_value.append(np.sum(next_state_rewards))

            value_table[state] = max(Q_value)

        threshold = 1e-20
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print(f'Value-iteration converged at iteration {i+1}')
            break
    return value_table

optimal_value_function = value_iteration(env=env, gamma=1.0)

# calculating optimal policy from optimal_value_function
def extract_policy(value_table, gamma=1.0):
    # first define random policy
    policy = np.zeros(env.observation_space.n)
    
    # then for each state we build Q_table
    # and for each action in that state compute Q value 
    # and add it to our Q_table
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + (gamma * value_table[next_state])))
        policy[state] = np.argmax(Q_table)
    return policy


optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(f"Optimal policy using value iteration {optimal_policy}")




######################################
##    Policy Iteration function     ## 
######################################

def compute_value_function(policy, gamma = 1.0):
    # We initialize value_table as zero with the number of states
    value_table = np.zeros(env.nS)
    threshold = 1e-10

    # for each state, we get the action from the policy
    # then we compute the value function 
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.nS):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state]) for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break 
    return value_table


# policy iteration
def policy_iteration(env, gamma = 0.1):
    # first initialize random_policyt as zero
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    gamma = 0.1
    for i in range(no_of_iterations):
        new_value_function = compute_value_function(random_policy, gamma)
        new_policy = extract_policy(new_value_function, gamma)
        if (np.all(random_policy == new_policy)):
            print(f"Policy-iteration converged at step {i+1}")
            break
        random_policy = new_policy
    return new_policy 


optimal_policy = policy_iteration(env, gamma = 0.1)
print(optimal_policy)



































