# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
To implement the Value Iteration algorithm using the Gymnasium FrozenLake-v1 environment to find the optimal state-value function and optimal policy. The algorithm uses the Bellman optimality equation to calculate the maximum expected reward for each state.





## Software Requirements
```
Python 
Gymnasium Library
NumPy
Jupyter Notebook/Google colab
```
## Environment Description
FrozenLake-v1 is a grid-based environment in Gymnasium where an agent moves on a frozen lake to reach the goal.

The environment consists of:
- S – Start: Starting position of the agent.
- F – Frozen: Safe surface to walk.
- H – Hole: Dangerous position; the episode ends.
- G – Goal: Destination where the agent receives a reward.
The agent can move in four directions: Left, Down, Right, and Up.

## MDP Representation
The FrozenLake environment is represented as a Markov Decision Process (MDP).
- States: 16 states (4 × 4 grid).
- Actions: 4 actions (Left, Down, Right, Up).
- Rewards: Reward of 1 for reaching the goal; otherwise 0.
- Transition Probability: Probability of moving from one state to another.
- Discount Factor: 0.99.

## Theory
Value Iteration is a Dynamic Programming algorithm used to find the optimal policy in a Markov Decision Process.

It repeatedly updates the value of each state using the Bellman Optimality Equation until the values become stable.

### Bellman Optimality Equation:
$$
V(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
$$

**Where:**
- `V(s)` – Value of the current state.
- `P(s'|s,a)` – Transition probability.
- `R(s,a,s')` – Immediate reward.
- `γ` – Discount factor.
- `a` – Possible action.
- `s'` – Next state.


## Algorithm
1. Initialize all state values to zero.
2. Set the number of states and actions.
3. Calculate the expected value for each action.
4. Update the state value using the maximum action value.
5. Calculate the difference between old and new values.
6. Repeat until the difference is smaller than the threshold.
7. Extract the optimal policy using the maximum action value.
8. Display the optimal state-value function and policy.




## Python Program

```python

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
def value_iteration(env, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))

        for s in range(n_states):
            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            policy[s] = np.argmax(action_values)

        iteration += 1

        if delta < theta:
            break

    return V, policy, iteration





```

---

## Output

```text

Name:     Sanjeev A      
Register Number:     212224230246      
Value Iteration Completed
Number of Iterations: 436

Optimal State-Value Function:
[[0.6494 0.5976 0.5639 0.5473]
 [0.669  0.     0.4672 0.    ]
 [0.709  0.7704 0.8518 0.    ]
 [0.     0.7738 0.8007 0.    ]]

Optimal Policy:
[['L' 'U' 'U' 'U']
 ['L' 'L' 'L' 'L']
 ['U' 'D' 'D' 'L']
 ['L' 'R' 'L' 'L']]
```

---

## Result
```text
The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

The optimal state-value function and optimal policy were computed using the Bellman optimality equation. The algorithm converged after 436 iterations.

```
---

## Inference
```text
Value Iteration successfully finds the optimal policy by repeatedly updating state values. The agent learns the best path to reach the goal while avoiding holes. The experiment demonstrates how Dynamic Programming can solve an MDP using known transition probabilities and rewards.

```
---

