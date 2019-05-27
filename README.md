# Reinforcement-learning-exercises-with-OpenAI-Gym

A recap on RL used fo this exercises with the [cheatsheet](cheatsheet.pdf). 

More theory about Reinforcement Learning can be find [here](https://mitpress.mit.edu/books/reinforcement-learning-second-edition).


## Temporal difference methods

Implementation of Sarsa, Q-Learning and Expected Sarsa in order to solve the [CliffWalking environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py).

`Temporal_Difference_Methods.ipynb` : Implementation of the three methods.

`plot_utils.py` : contains a plotting function for visualizing state-value functions and policies.

`check_test.py` : contains unit tests to check the validity of your implementations.

![alt text](Images/CliffWalking.JPG)



## OpenAI Gym Taxi-v2 with Temporal difference methods

Work based on the part 3.1 of this [paper](https://arxiv.org/pdf/cs/9905014.pdf) to solve the [Taxi-v2 environment](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).

`agent.py`: The reinforcement learning agent is developed here.

`monitor.py`: The interact function tests how well your agent learns from interaction with the environment.

`main.py`: Run this file in the terminal to check the performance of your agent.

When you run `main.py`, the agent that specify in `agent.py` interacts with the environment for 20,000 episodes. The details of the interaction are specified in `monitor.py`, which returns two variables: avg_rewards and best_avg_reward. The best_avg_reward is used to see how well the agent performed in the task.

![alt text](Images/taxi-v2.png)

