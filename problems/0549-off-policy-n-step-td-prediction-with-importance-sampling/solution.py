import numpy as np

def off_policy_nstep_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    n: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """
    Off-policy n-step TD prediction for state values using importance sampling.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        n: Number of steps for the n-step return.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        V: numpy array of shape (num_states,) with estimated state values.
    """
    V = np.zeros(num_states)
    for episode in episodes:
        
        T = len(episode)
        for t in range(T):
            rho = 1.0
            for k in range(t, min(t+n, T)):
                state_k, action_k, reward_k = episode[k]
                pi_prob = target_policy[state_k][action_k]
                b_prob = behavior_policy[state_k][action_k]

                if b_prob == 0:
                    rho = 0
                    break
                
                rho *= pi_prob / b_prob
            
            if rho == 0:
                continue
            
            G = 0.0
            for i in range(min(n, T-t)):
                state, action, reward = episode[t+i]
                G += (gamma ** i) * reward
            
            if t + n < T:
                next_state = episode[t+n][0]
                G += (gamma ** n) * V[next_state]

            state_t = episode[t][0]
            V[state_t] += alpha * rho * (G - V[state_t]) 

            


    return V