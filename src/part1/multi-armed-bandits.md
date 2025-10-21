# Multi-armed Bandits

In reinforcement learning problems are reduced to an environment and agent. The agent
makes observations of the environment which are used to determine the current state
and reward for being in the state. The agent's goal is to use the state and reward
to select an action that leads the agent to a new state with higher reward.

Multi-armed bandits are the simplest case of agent-environment problem. There is no state,
each turn the agent will select an action (which lever to pull on the slot machine). This
is the simplest case because there are no states, only actions and rewards.

This demonstrates a key dilemma between exploration and exploitation. As the agent pulls levers
and recieves rewards it builds a probability distribution of the reward for each action.
As the agent becomes more confident it should explore less and exploit what it knows.

The optimal action-value function is defined as @@Sutton2018ReinforcementLearning
$$ q_*(a) = E[R_t \mid A_t = a] $$
