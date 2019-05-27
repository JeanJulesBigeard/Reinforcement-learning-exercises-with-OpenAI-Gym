import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=1,decay=0.999, gamma=0.9,alpha=0.1):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_function = None;
        self.next_value_function = None;
        
        ## With current settings: Best average reward 9.291
        #self.init_for_sarsa()
        
        ## With current settings:  Best average reward 9.213
        #self.init_for_q_learning()
        
        ## With current settings:  Best average reward 9.185
        self.init_for_expected_sarsa()
        
    def init_for_sarsa(self,epsilon = 1,decay = 0.999,gamma = 0.9,alpha = 0.1):
        self.epsilon = epsilon
        self.decay = decay 
        self.gamma = gamma 
        self.alpha = alpha
        self.epsilon_function = self.decayed_epsilon
        def calculate(next_state):
            return self.Q[next_state][self.select_action(next_state)]
        self.next_value_function = calculate 
        
    def init_for_q_learning(self,epsilon = 1,decay = 0.9,gamma = 0.9,alpha = 0.2):
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma    
        self.alpha = alpha
        self.epsilon_function = self.decayed_min_epsilon
        def calculate(next_state):
            return np.max(self.Q[next_state])
        self.next_value_function = calculate
        
    def init_for_expected_sarsa(self,epsilon = 0.01,decay = 0.9,gamma = 0.8,alpha = 0.65):
        self.epsilon = epsilon
        self.decay = decay
        #discounted return rate
        self.gamma = gamma
        #tendency to use recent event
        self.alpha = alpha
        self.epsilon_function = self.fixed_epsilon
        def calculate(next_state):
            policy_s = self.get_action_probability(next_state)
            next_action_value = np.dot(self.Q[next_state], policy_s)
            return np.max(self.Q[next_state])
        self.next_value_function = calculate
        
    def decayed_min_epsilon(self):
        self.epsilon = max(self.epsilon*self.decay,0.01)
        
    def decayed_epsilon(self):
        self.epsilon *= self.decay
        
    def fixed_epsilon(self):
        pass
    
    def get_action_probability(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA  
        best_action = np.argmax(self.Q[state])
        policy_s[best_action] += 1 - self.epsilon
        return policy_s
    
    def select_action(self, state):
        policy_s = self.get_action_probability(state)
        return np.random.choice(np.arange(self.nA), p=policy_s)
    
    def step(self, state, action, reward, next_state, done):
        self.epsilon_function()
        next_action_value = 0
        if not done:
            next_action_value = self.next_value_function(next_state)
        self.Q[state][action] += self.alpha * (reward  + self.gamma * next_action_value - self.Q[state][action])