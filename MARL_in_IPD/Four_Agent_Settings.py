# This is the main file of the implementation of single-agent setting Iterated Prisoner's Dilemma game: Agent VS ALLD or ALLC
# The Play Class uses the PrisonersDilemma Class (Environment in 'game_class' python file)
# and the Simple_Policy_Gradient Class (simple policy gradient neural network).
import os
import sys

sys.path.append(os.getcwd())
import pandas as pd
import torch.optim as optim
import itertools
import json
import random

# import relevant classes from other modules in this project:
from Simple_Policy_Gradient_Agent import *
from Prisoners_Dilemma_Game import *


class Play:

    # Format of dictionary: [R,S,T,P,GAMMA, DELTA, LEARNING_RATE, my_seed,
    # episode, game, agent_one, agent_two]
    def __init__(self, parameters):

        self.r = parameters["R"]
        self.s = parameters["S"]
        self.t = parameters["T"]
        self.p = parameters["P"]
        self.GAMMA = parameters["Gamma"]
        self.DELTA = parameters["Delta"]
        self.LEARNING_RATE = parameters["Learning_rate"]
        self.my_seed = parameters["Seed"]
        self.file_name = parameters["Output_filename"]
        self.episodes = parameters["Episodes"]
        self.runs = parameters["Runs"]

        # Manually seed for make experiments reproducible:
        np.random.seed(self.my_seed)
        torch.manual_seed(self.my_seed)

        game = PrisonersDilemma
        # Create an object of the environment class:
        self.env = game(self.r, self.s, self.t, self.p, self.my_seed)

        # create agent objects:
        #Agent One
        self.agent_one = Simple_Policy_Gradient(self.env.STATE_NUM,
                                           self.env.ACTION_NUM,
                                           self.my_seed)
        #Agent Two
        self.agent_two = Simple_Policy_Gradient(self.env.STATE_NUM,
                                           self.env.ACTION_NUM,
                                           self.my_seed)
        #Agent Three
        self.agent_three = Simple_Policy_Gradient(self.env.STATE_NUM,
                                           self.env.ACTION_NUM,
                                           self.my_seed)
        #Agent Four
        self.agent_four = Simple_Policy_Gradient(self.env.STATE_NUM,
                                           self.env.ACTION_NUM,
                                           self.my_seed)

        #create optimizer objects:
        #Optimizer one
        self.optimizer_agent_one = optim.Adam(self.agent_one.parameters(),
                                        lr=self.LEARNING_RATE)
        #Optimizer Two
        self.optimizer_agent_two = optim.Adam(self.agent_two.parameters(),
                                        lr=self.LEARNING_RATE)
        #Optimizer Three
        self.optimizer_agent_three = optim.Adam(self.agent_three.parameters(),
                                        lr=self.LEARNING_RATE)
        #Optimizer Four
        self.optimizer_agent_four = optim.Adam(self.agent_four.parameters(),
                                        lr=self.LEARNING_RATE)


    def gaming(self):
        # this method is the main method of Play class which plays many IPDs

        # LOSS LISTS FOR PLOTTING
        agent_one_loss_history = []
        agent_two_loss_history = []
        agent_three_loss_history = []
        agent_four_loss_history = []

        # AVERAGE CUMULATIE REWARDS LISTS FOR PLOTTING:
        agent_one_avg_cum_rew = []
        agent_two_avg_cum_rew = []
        agent_three_avg_cum_rew = []
        agent_four_avg_cum_rew = []

        # LIST OF PROBABABILITY OF COOPERATION
        agent_one_first = []  # First ever round
        agent_two_first = []
        agent_three_first = []
        agent_four_first = []

        agent_one_r = []  # Reward
        agent_two_r = []
        agent_three_r = []
        agent_four_r = []

        agent_one_s = []  # Sucker
        agent_two_s = []
        agent_three_s = []
        agent_four_s = []

        agent_one_t = []  # Temptation
        agent_two_t = []
        agent_three_t = []
        agent_four_t = []

        agent_one_p = []  # Punishment
        agent_two_p = []
        agent_three_p = []
        agent_four_p = []

        episode_count = []

        for i in range(self.runs):
            initial_state = self.env.reset()  # resets the environment at the beginnning

            # This loop creates empty buffers at the start of game and resets
            # these every 2000 IPDs completed (after a training session is
            # done).
            if (i == 0) or (i % self.episodes == 0):
                # for AGENT ONE - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_agent_one, batch_actions_agent_one, batch_discounted_rewards_agent_one = [], [], []
                current_rewards_agent_one = []

                # for AGENT TWO - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_agent_two, batch_actions_agent_two, batch_discounted_rewards_agent_two = [], [], []
                current_rewards_agent_two = []

                # for AGENT THREE - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_agent_three, batch_actions_agent_three, batch_discounted_rewards_agent_three = [], [], []
                current_rewards_agent_three = []

                # for AGENT FOUR - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_agent_four, batch_actions_agent_four, batch_discounted_rewards_agent_four = [], [], []
                current_rewards_agent_four = []
                # Who is playing against who?

                # Who plays against who?

                list_of_players = list(range(4))
                random_list_agents = random.sample(list_of_players, len(list_of_players))

                permutation_result = itertools.permutations(random_list_agents, 2)
                for match in permutation_result:
                    #print(match)
                    player_one_index = match[0]
                    player_two_index = match[1]

                    while self.is_continue(self):  # Should we continue to the next round?

                        actions = []  # temporary list of actions for a round
                        # PREDICTION

                        # AGENT: getting the action and its probability from
                        # predict function
                        # getting the action and its probability from predict function
                        if player_one_index == 0:
                            action_player_one, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_one)
                            # add agent one's action in a buffer
                            batch_actions_agent_one.append(int(action_player_one))

                        elif player_one_index == 1:
                            action_player_one, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_two)
                            # add agent two's action in a buffer
                            batch_actions_agent_two.append(int(action_player_one))

                        elif player_one_index == 2:
                            action_player_one, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_three)
                            # add agent two's action in a buffer
                            batch_actions_agent_three.append(int(action_player_one))

                        elif player_one_index == 3:
                            action_player_one, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_four)
                            # add agent two's action in a buffer
                            batch_actions_agent_four.append(int(action_player_one))

                        actions.append(action_player_one)

                        if player_two_index == 0:
                            action_player_two, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_one)
                            # add agent one's action in a buffer
                            batch_actions_agent_one.append(int(action_player_two))

                        elif player_two_index == 1:
                            action_player_two, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_two)
                            # add agent one's action in a buffer
                            batch_actions_agent_two.append(int(action_player_two))

                        elif player_two_index == 2:
                            action_player_two, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_three)
                            # add agent one's action in a buffer
                            batch_actions_agent_three.append(int(action_player_two))

                        elif player_two_index == 3:
                            action_player_two, action_probability = self.predict(
                                                                        initial_state,
                                                                        self.agent_four)
                            # add agent one's action in a buffer
                            batch_actions_agent_four.append(int(action_player_two))

                        actions.append(action_player_two)

                        # PLAY THE GAME
                        # Get the state, reward of both agents (as a list) and step
                        # from environment class:
                        state, reward_both_agents, round_counter = self.env.step(actions)

                        # Separate the rewards for each agent:
                        reward_player_one = reward_both_agents[0]
                        reward_player_two = reward_both_agents[1]

                        # Add state and reward of player one:
                        if player_one_index == 0:
                            ### Update the buffer for AGENT ONE ###
                            # add the current state in a buffer
                            batch_states_agent_one.append(state)
                            # add agent one's reward in a buffer
                            current_rewards_agent_one.append(reward_player_one)

                        elif player_one_index == 1:
                            ### Update the buffer for AGENT TWO ###
                            # add the current state in a buffer
                            batch_states_agent_two.append(state)
                            # add agent one's reward in a buffer
                            current_rewards_agent_two.append(reward_player_one)

                        elif player_one_index == 2:

                            batch_states_agent_three.append(state)

                            current_rewards_agent_three.append(reward_player_one)

                        elif player_one_index == 3:
                            batch_states_agent_four.append(state)

                            current_rewards_agent_four.append(reward_player_one)

                        # Add state and reward of player two:
                        if player_two_index == 0:
                            ### Update the buffer for AGENT ONE ###
                            # add the current state in a buffer
                            batch_states_agent_one.append(state)
                            # add agent one's reward in a buffer
                            current_rewards_agent_one.append(reward_player_two)

                        elif player_two_index == 1:
                            batch_states_agent_two.append(state)

                            current_rewards_agent_two.append(reward_player_two)

                        elif player_two_index == 2:
                            batch_states_agent_three.append(state)

                            current_rewards_agent_three.append(reward_player_two)

                        elif player_two_index == 3:
                            batch_states_agent_four.append(state)

                            current_rewards_agent_four.append(reward_player_two)

                        # update the state of the environment for next round.
                        initial_state = state

            if (i != 0) and (i % self.episodes == 0):
                # For average cumulative reward graphs:
                # CUMULATIVE REWARDS PER EPISODE:
                # calculate sum of rewards of all rounds in an episode for each
                # agent
                agent_one_sum_cur_rew = sum(current_rewards_agent_one)
                agent_two_sum_cum_rew = sum(current_rewards_agent_two)
                agent_three_sum_cum_rew = sum(current_rewards_agent_three)
                agent_four_sum_cum_rew = sum(current_rewards_agent_four)

                # AVERAGE CUMULATIVE REWARD
                # divide the cumulative reward per episode by total number of
                # rounds for each agent
                agent_one_avg_rew = agent_one_sum_cur_rew / round_counter
                agent_two_avg_rew = agent_two_sum_cum_rew / round_counter
                agent_three_avg_rew = agent_three_sum_cum_rew / round_counter
                agent_four_avg_rew = agent_four_sum_cum_rew / round_counter

                agent_one_avg_cum_rew.append(agent_one_avg_rew)
                agent_two_avg_cum_rew.append(agent_two_avg_rew)
                agent_three_avg_cum_rew.append(agent_three_avg_rew)
                agent_four_avg_cum_rew.append(agent_four_avg_rew)

                ### DISCOUNTED REWARD CALCULATION ###
                # Discounted reward calculation for each agent one and addition of
                # that information to a buffer.

                # Agent One
                batch_discounted_rewards_agent_one.extend(
                    self.discounted_rewards(
                        (current_rewards_agent_one), GAMMA=self.GAMMA))
                # Agent Two
                batch_discounted_rewards_agent_two.extend(
                    self.discounted_rewards(
                        (current_rewards_agent_two), GAMMA=self.GAMMA))
                # Agent Three
                batch_discounted_rewards_agent_three.extend(
                    self.discounted_rewards(
                        (current_rewards_agent_three), GAMMA=self.GAMMA))
                # Agent Four
                batch_discounted_rewards_agent_four.extend(
                    self.discounted_rewards(
                        (current_rewards_agent_four), GAMMA=self.GAMMA))

                # CONVERSION OF BUFFER LISTS INTO TENSORS
                # Convert states, actions and discounted_rewards buffers from
                # lists to tensors for each agent:

                batch_states_t_agent_one, batch_actions_t_agent_one, batch_reward_discounted_t_agent_one = self.tensor_conversion(
                    batch_states_agent_one,
                    batch_actions_agent_one,
                    batch_discounted_rewards_agent_one)

                batch_states_t_agent_two, batch_actions_t_agent_two, batch_reward_discounted_t_agent_two = self.tensor_conversion(
                    batch_states_agent_two,
                    batch_actions_agent_two,
                    batch_discounted_rewards_agent_two)

                batch_states_t_agent_three, batch_actions_t_agent_three, batch_reward_discounted_t_agent_three = self.tensor_conversion(
                    batch_states_agent_three,
                    batch_actions_agent_three,
                    batch_discounted_rewards_agent_three)

                batch_states_t_agent_four, batch_actions_t_agent_four, batch_reward_discounted_t_agent_four = self.tensor_conversion(
                    batch_states_agent_four,
                    batch_actions_agent_four,
                    batch_discounted_rewards_agent_four)

                # TRAINING

                # AGENT ONE
                # Call the train function to train agent one and get the loss tensor
                loss_agent_one = self.train(self.agent_one,
                                            self.optimizer_agent_one,
                                            batch_states_t_agent_one,
                                            batch_reward_discounted_t_agent_one,
                                            batch_actions_t_agent_one)

                loss_value_agent_one = loss_agent_one.item()  # extracting the loss value from tensor

                agent_one_loss_history.append(loss_value_agent_one)  # appending loss value to the list

                # PROBABILITY OF COOPERATION DEPENDING ON STATE

                # First Ever Round:
                agent_one_prob_first = self.get_single_state_strategy(agent=self.agent_one, state="o")
                agent_one_first.append(agent_one_prob_first)
                # R
                agent_one_prob_r = self.get_single_state_strategy(agent=self.agent_one, state="r")
                agent_one_r.append(agent_one_prob_r)
                # S
                agent_one_prob_s = self.get_single_state_strategy(agent=self.agent_one, state="s")
                agent_one_s.append(agent_one_prob_s)
                # T
                agent_one_prob_t = self.get_single_state_strategy(agent=self.agent_one, state="t")
                agent_one_t.append(agent_one_prob_t)
                # P
                agent_one_prob_p = self.get_single_state_strategy(agent=self.agent_one, state="p")
                agent_one_p.append(agent_one_prob_p)

                # AGENT TWO
                # Call the train function to train agent one and get the loss tensor
                loss_agent_two = self.train(self.agent_two,
                                            self.optimizer_agent_two,
                                            batch_states_t_agent_two,
                                            batch_reward_discounted_t_agent_two,
                                            batch_actions_t_agent_two)

                loss_value_agent_two = loss_agent_two.item()  # extracting the loss value from tensor

                agent_two_loss_history.append(loss_value_agent_two)  # appending loss value to the list

                # PROBABILITY OF COOPERATION DEPENDING ON STATE
                agent_two_prob_first = self.get_single_state_strategy(agent=self.agent_two, state="o")
                agent_two_first.append(agent_two_prob_first)
                # R
                agent_two_prob_r = self.get_single_state_strategy(agent=self.agent_two, state="r")
                agent_two_r.append(agent_two_prob_r)
                # S
                agent_two_prob_s = self.get_single_state_strategy(agent=self.agent_two, state="s")
                agent_two_s.append(agent_two_prob_s)
                # T
                agent_two_prob_t = self.get_single_state_strategy(agent=self.agent_two, state="t")
                agent_two_t.append(agent_two_prob_t)
                # P
                agent_two_prob_p = self.get_single_state_strategy(agent=self.agent_two, state="p")
                agent_two_p.append(agent_two_prob_p)

                # AGENT THREE
                # Call the train function to train agent one and get the loss tensor
                loss_agent_three = self.train(self.agent_three,
                                              self.optimizer_agent_three,
                                              batch_states_t_agent_three,
                                              batch_reward_discounted_t_agent_three,
                                              batch_actions_t_agent_three)

                loss_value_agent_three = loss_agent_three.item()  # extracting the loss value from tensor

                agent_three_loss_history.append(loss_value_agent_three)  # appending loss value to the list

                # PROBABILITY OF COOPERATION DEPENDING ON STATE
                agent_three_prob_first = self.get_single_state_strategy(agent=self.agent_three, state="o")
                agent_three_first.append(agent_three_prob_first)
                # R
                agent_three_prob_r = self.get_single_state_strategy(agent=self.agent_three, state="r")
                agent_three_r.append(agent_three_prob_r)
                # S
                agent_three_prob_s = self.get_single_state_strategy(agent=self.agent_three, state="s")
                agent_three_s.append(agent_three_prob_s)
                # T
                agent_three_prob_t = self.get_single_state_strategy(agent=self.agent_three, state="t")
                agent_three_t.append(agent_three_prob_t)
                # P
                agent_three_prob_p = self.get_single_state_strategy(agent=self.agent_three, state="p")
                agent_three_p.append(agent_three_prob_p)

                # AGENT FOUR
                # Call the train function to train agent one and get the loss tensor
                loss_agent_four = self.train(self.agent_four,
                                             self.optimizer_agent_four,
                                             batch_states_t_agent_four,
                                             batch_reward_discounted_t_agent_four,
                                             batch_actions_t_agent_four)

                loss_value_agent_four = loss_agent_four.item()  # extracting the loss value from tensor

                agent_four_loss_history.append(loss_value_agent_four)  # appending loss value to the list

                # PROBABILITY OF COOPERATION DEPENDING ON STATE
                agent_four_prob_first = self.get_single_state_strategy(agent=self.agent_four, state="o")
                agent_four_first.append(agent_four_prob_first)
                # R
                agent_four_prob_r = self.get_single_state_strategy(agent=self.agent_four, state="r")
                agent_four_r.append(agent_four_prob_r)
                # S
                agent_four_prob_s = self.get_single_state_strategy(agent=self.agent_four, state="s")
                agent_four_s.append(agent_four_prob_s)
                # T
                agent_four_prob_t = self.get_single_state_strategy(agent=self.agent_four, state="t")
                agent_four_t.append(agent_four_prob_t)
                # P
                agent_four_prob_p = self.get_single_state_strategy(agent=self.agent_four, state="p")
                agent_four_p.append(agent_four_prob_p)

                # An index of episodes:
                episode_count.append(i)

            # This loop is for training the agent after many IPDs

            i = + 1
        self.output_as_excel(episode_count,
                                 agent_one_avg_cum_rew,
                                 agent_two_avg_cum_rew,
                                 agent_three_avg_cum_rew,
                                 agent_four_avg_cum_rew,
                                 agent_one_loss_history,
                                 agent_two_loss_history,
                                 agent_three_loss_history,
                                 agent_four_loss_history,
                                 agent_one_first,
                                 agent_two_first,
                                 agent_three_first,
                                 agent_four_first,
                                 agent_one_r,
                                 agent_two_r,
                                 agent_three_r,
                                 agent_four_r,
                                 agent_one_s,
                                 agent_two_s,
                                 agent_three_s,
                                 agent_four_s,
                                 agent_one_t,
                                 agent_two_t,
                                 agent_three_t,
                                 agent_four_t,
                                 agent_one_p,
                                 agent_two_p,
                                 agent_three_p,
                                 agent_four_p,
                                 self.file_name)







    def output_as_excel(self, episode_index,
                        avg_cum_reward_one,
                        avg_cum_reward_two,
                        avg_cum_reward_three,
                        avg_cum_reward_four,
                        loss_value_one,
                        loss_value_two,
                        loss_value_three,
                        loss_value_four,
                        agent_one_first,
                        agent_two_first,
                        agent_three_first,
                        agent_four_first,
                        agent_one_r,
                        agent_two_r,
                        agent_three_r,
                        agent_four_r,
                        agent_one_s,
                        agent_two_s,
                        agent_three_s,
                        agent_four_s,
                        agent_one_t,
                        agent_two_t,
                        agent_three_t,
                        agent_four_t,
                        agent_one_p,
                        agent_two_p,
                        agent_three_p,
                        agent_four_p,
                        file_name):

        # Creating some dictionaries that can store global variables:
        GAMMA = [0]
        DELTA = [0]
        LEARNING_RATE = [0]
        SEED = [0]
        EPISODE = [0]
        RUNS = [0]
        R = [0]
        S = [0]
        T = [0]
        P = [0]

        # Appending relevant information to appropriate dictionaries:
        GAMMA[0] = self.GAMMA
        DELTA[0] = self.DELTA
        LEARNING_RATE[0] = self.LEARNING_RATE
        SEED[0] = self.my_seed
        EPISODE[0] = self.episodes
        RUNS[0] = self.runs
        R[0] = self.r
        S[0] = self.s
        T[0] = self.t
        P[0] = self.p

        df_agent_one = pd.DataFrame({
            'Episode': episode_index,
            'Agent_One_Avg_Cumulative_Reward': avg_cum_reward_one,
            'Agent_One_Loss_Value': loss_value_one,
            'Agent_One_Prob_First_Round': agent_one_first,
            'Agent_One_Prob_P': agent_one_p,
            'Agent_One_Prob_R': agent_one_r,
            'Agent_One_Prob_S': agent_one_s,
            'Agent_One_Prob_T': agent_one_t

        })

        df_agent_two = pd.DataFrame({
            'Episode': episode_index,
            'Agent_Two_Avg_Cumulative_Reward': avg_cum_reward_two,
            'Agent_Two_Loss_Value': loss_value_two,
            'Agent_Two_Prob_First_Round': agent_two_first,
            'Agent_Two_Prob_P': agent_two_p,
            'Agent_Two_Prob_R': agent_two_r,
            'Agent_Two_Prob_S': agent_two_s,
            'Agent_Two_Prob_T': agent_two_t

        })

        df_agent_three = pd.DataFrame({
            'Episode': episode_index,
            'Agent_Three_Avg_Cumulative_Reward': avg_cum_reward_three,
            'Agent_Three_Loss_Value': loss_value_three,
            'Agent_Three_Prob_First_Round': agent_three_first,
            'Agent_Three_Prob_P': agent_three_p,
            'Agent_Three_Prob_R': agent_three_r,
            'Agent_Three_Prob_S': agent_three_s,
            'Agent_Three_Prob_T': agent_three_t

        })

        df_agent_four = pd.DataFrame({
            'Episode': episode_index,
            'Agent_Four_Avg_Cumulative_Reward': avg_cum_reward_four,
            'Agent_Four_Loss_Value': loss_value_four,
            'Agent_Four_Prob_First_Round': agent_four_first,
            'Agent_Four_Prob_P': agent_four_p,
            'Agent_Four_Prob_R': agent_four_r,
            'Agent_Four_Prob_S': agent_four_s,
            'Agent_Four_Prob_T': agent_four_t

        })


        df_detail = pd.DataFrame({
            'GAMMA': GAMMA,
            'DELTA': DELTA,
            'LEARNING_RATE': LEARNING_RATE,
            'NUMBER_OF_EPISODES': EPISODE,
            'RUNS': RUNS,
            'SEED': SEED,
            'R': R,
            'S': S,
            'T': T,
            'P': P,
        })

        name_of_file = file_name

        with pd.ExcelWriter(name_of_file) as writer:
            df_agent_one.to_excel(writer, sheet_name="Agent_One")
            df_agent_two.to_excel(writer, sheet_name="Agent_Two")
            df_agent_three.to_excel(writer, sheet_name="Agent_Three")
            df_agent_four.to_excel(writer, sheet_name="Agent_Four")
            df_detail.to_excel(writer, sheet_name="information")


    def discounted_rewards(self, rewards, GAMMA):
        # this method calculates the discounted rewards
        # @param: list of rewards of rounds for an agent, value of gamma
        # @returns: list (of discounted rewards)
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= GAMMA
            sum_r += r
            res.append(sum_r)

        return list(reversed(res))

    def predict(self, state, agent):

        current_environment_state = np.asarray(state)

        # Convert numpy array to tensor:
        current_environment_tensor_state = torch.from_numpy(
            current_environment_state).type(torch.FloatTensor)

        # Get action probability which is the output from neural network:
        action_probability = agent(current_environment_tensor_state)

        # First detach the grad and then get the probability as a numpy array:
        action_probability_array = action_probability.detach().numpy()

        prob_defection = action_probability_array[0][0]

        random_number = np.random.random()

        if random_number < prob_defection:
            action = 0
        else:
            action = 1

        return action, action_probability

    def tensor_conversion(self, state_buffer, action_buffer, qvals_buffer):

        # Converting the buffers into tensors:

        batch_states = state_buffer
        batch_actions = action_buffer
        batch_qvals = qvals_buffer

        # State Buffer:
        # convert state buffer to FloatTensor
        batch_states_tensor = torch.FloatTensor(batch_states)

        # Action Buffer:
        # convert actions buffer to LongTensor
        batch_actions_tensor = torch.LongTensor(batch_actions)

        # Discounted Rewards Buffer:
        # convert discounted reward list to FloatTensor
        batch_qvals_tensor = torch.FloatTensor(batch_qvals)

        return batch_states_tensor, batch_actions_tensor, batch_qvals_tensor

    def train(
            self,
            agent,
            optimizer,
            batch_states_t,
            batch_qvals_t,
            batch_actions_t):

        batch_states = batch_states_t
        batch_qvals = batch_qvals_t
        batch_actions = batch_actions_t

        optimizer_agent = optimizer

        # Clearing our the gradients before training the agent:
        optimizer_agent.zero_grad()  # clear out the gradients

        # Loss Calculation:

        # Step 1: Calculate the probability of the actions actually taken in
        # each round in an episode:
        # get the probs from the NN by sending STATES tensor through agent
        logits_v = agent(batch_states)

        # Intermediate step - reshaping tensor before multiplication:
        # transpose the output from Neural network of agent one
        log_prob_v = logits_v.transpose(0, -1).contiguous()

        # convert the 3-D tensor to 2-D tensor
        log_prob_v = log_prob_v.view(log_prob_v.size(0), -1)

        # transpose the resultant vector again for matrix multiplication
        log_prob_v = log_prob_v.transpose(0, -1).contiguous()

        # Extract just the predicted probability for the action that was
        # actually taken:
        log_prob_v_intermediate = log_prob_v[range(
            len(batch_states)), batch_actions]

        # Step 2: Calculate the log of the extracted probability for the
        # actions actually taken
        log_prob_v_intermediate_log = torch.log(log_prob_v_intermediate)

        # Step 3: Multiply it by discounted reward:
        log_prob_actions_v = batch_qvals * log_prob_v_intermediate_log

        # Step 4: Calculate the mean loss and flip the sign of loss value
        # find the mean of the above to calculation to determine loss
        loss_v = -1 * log_prob_actions_v.mean()

        # Step 5: Backpropagate and minimize the loss:
        # run the backwards() function to do the required gradient calculations
        # and so on.
        loss_v.backward()
        optimizer_agent.step()  # take a step towards minimizing the loss

        return loss_v

    def game_reset(self):

        initial_state = self.env.reset()
        return initial_state

    @staticmethod
    def is_continue(self):

        # checks if the Prisoner's Dilemma game should continue after each round
        # continuity of probability, delta, is used.
        # @returns: a boolean

        game_continue = False
        if (self.env.step_count == 0) or (
                (self.env.step_count != 0) and (np.random.rand() < self.DELTA)):
            game_continue = True

        return game_continue

    def prob_cooperate(self, action_prob):
        # This method extracts the probability to cooperate from the output of
        # the neural network
        # detach the grad information from the output of NN
        action_prob = action_prob.detach().numpy()

        action_prob_array = np.asarray(action_prob)  # convert into an array

        cooperate_prob = action_prob_array[0][1]

        return cooperate_prob

    def get_single_state_strategy(self, agent, state):

        input_state = state
        action, action_prob = self.predict(
            state=(self.get_states(input_state)), agent=agent)  # fist round
        cooperate_prob = self.prob_cooperate(action_prob)  # first round

        return cooperate_prob

    def get_states(self, state):
        # pass letter "o"  to get initial state
        # pass letter "r"  to get Reward state (both agents cooperate)
        # pass letter "s"  to get Sucker state (agent ONE cooperates and TWO defection)
        # pass letter "t"  to get Temptation (agent ONE defection and agent TWO cooperation)
        # pass letter "p"  to get Punishment (agent ONE and agent TWO both
        # defects)

        if state == "o":
            # initial state when game begins:
            current_state = self.game_reset()
        elif state == "r":
            # R
            current_state, reward, counter = self.env.step(action=[1, 1])
        elif state == "s":
            # S
            current_state, reward, counter = self.env.step(action=[1, 0])
        elif state == "t":
            # T
            current_state, reward, counter = self.env.step(action=[0, 1])
        elif state == "p":
            # P
            current_state, reward, counter = self.env.step(action=[0, 0])
        else:
            print("wrong string entered")

        return current_state



def print_example_json():
    conf = dict()
    conf["R"] = 4
    conf["S"] = 0
    conf["T"] = 6
    conf["P"] = 2
    conf["Gamma"] = 0.99
    conf["Delta"] = 0.8
    conf["Learning_rate"] = 0.0001
    conf["Seed"] = 123456789
    conf["Output_filename"] = "4agents_lr0.0001_1000ep_123456789.xlsx"
    conf["Episodes"] = 1000
    conf["Runs"] = 1000000
    print(json.dumps(conf, indent=4))


def print_usage():
    print("Example 1: python Four_Agent_Settings.py ")
    print("will print an example json file")
    print()
    print("Example 2: python Four_Agent_Settings.py example.json")
    print("will run the simulation with the parameters in example.json")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_example_json()
    elif len(sys.argv) == 2:
        parameter_dictionary = json.loads(open(sys.argv[1]).read())
        game = Play(parameter_dictionary)
        game.gaming()

