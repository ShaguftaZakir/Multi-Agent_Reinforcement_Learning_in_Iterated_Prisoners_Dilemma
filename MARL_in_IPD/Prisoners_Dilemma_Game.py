import numpy as np
from gym.spaces import Discrete, Tuple
import sklearn.preprocessing as skl


class PrisonersDilemma():
    # This Class is a two-player Prisoner's Dilemma game which observation, reward and step_count (the round number)

    AGENT_NUM = 2
    ACTION_NUM = 2
    STATE_NUM = 5

    def __init__(self,r,s,t,p,my_seed,):
        self.action = tuple([Discrete(self.ACTION_NUM), Discrete(self.ACTION_NUM)])
        self.observation_space = tuple([skl.OneHotEncoder(self.STATE_NUM), skl.OneHotEncoder(self.STATE_NUM)]) #OneHot vectors for states
        self.payoff_mat = np.array([[p, s], [t, r]])  # the payoff matrix
        self.step_count = 0
        np.random.seed(my_seed)

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.STATE_NUM)
        init_state[-1] = 1  # this is because (0,0,0,0,1) represents initial state as we are using OneHot for observation space.
        observations = [init_state]
        return observations

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = [self.payoff_mat[ac1][ac0], self.payoff_mat[ac0][ac1]]

        state = np.zeros(self.STATE_NUM)
        state[ac0 * 2 + ac1] = 1
        observations = [state]  # we assume that from perspective of both agents, agent state is same.

        step_counter = self.step_count
        return observations, rewards, step_counter