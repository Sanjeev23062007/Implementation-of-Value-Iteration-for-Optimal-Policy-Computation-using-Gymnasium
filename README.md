# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement






## Software Requirements

## Environment Description


## MDP Representation


## Theory




## Algorithm





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
Write your result here

```
---

## Inference
```text
Write the inference here


```
---

