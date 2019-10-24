# This is the main file of the implementation of single-agent setting Iterated Prisoner's Dilemma game: Agent VS ALLD or ALLC
# The Play Class uses the PrisonersDilemma Class (Environment in 'game_class' python file)
# and the Simple_Policy_Gradient Class (simple policy gradient neural network).
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import torch.optim as optim
import json

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
        agent_one = Simple_Policy_Gradient
        agent_two = Simple_Policy_Gradient
        # create an object of the environment Class.
        self.env = game(self.r, self.s, self.t, self.p, self.my_seed)
        self.agent_one = agent_one(
            self.env.STATE_NUM,
            self.env.ACTION_NUM,
            self.my_seed)  # agent one
        self.agent_two = agent_two(
            self.env.STATE_NUM,
            self.env.ACTION_NUM,
            self.my_seed)  # agent two
        self.optimizer_agent_one = optim.Adam(
            self.agent_one.parameters(),
            lr=self.LEARNING_RATE)   # optimizer for agent one
        self.optimizer_agent_two = optim.Adam(
            self.agent_two.parameters(),
            lr=self.LEARNING_RATE)   # optimizer for agent one

    def gaming(self):
        # this method is the main method of Play class which plays many IPDs

        # LOSS LISTS FOR PLOTTING
        agent_one_loss_history = []
        agent_two_loss_history = []

        # AVERAGE CUMULATIE REWARDS LISTS FOR PLOTTING:
        agent_one_avg_cum_rew = []
        agent_two_avg_cum_rew = []

        # LIST OF PROBABABILITY OF COOPERATION
        agent_one_first = []   # First ever round
        agent_two_first = []

        agent_one_r = []  # Reward
        agent_two_r = []

        agent_one_s = []  # Sucker
        agent_two_s = []

        agent_one_t = []  # Temptation
        agent_two_t = []

        agent_one_p = []  # Punishment
        agent_two_p = []

        episode_count = []

        for i in range(self.runs):
            initial_state = self.env.reset()  # resets the environment at the beginnning

            # This loop creates empty buffers at the start of game and resets
            # these every 2000 IPDs completed (after a training session is
            # done).
            if (i == 0) or (i % self.episodes == 0):
                # for AGENT ONE - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_one, batch_actions_one, batch_qvals_agent_one = [], [], []
                cur_rewards_one = []

                # for AGENT TWO - Create some empty buffers for storing states, rewards and actions of all round of an episode.
                # These are reset at the start of a new episode.
                batch_states_two, batch_actions_two, batch_qvals_agent_two = [], [], []
                cur_rewards_two = []

            while self.is_continue(
                    self):  # Should we continue to the next round?

                ### PREDICTION ###

                # AGENT ONE: getting the action and its probability from
                # predict function
                # getting the action and its probability from predict function
                action_agent_one, action_probability_one = self.predict(
                    initial_state, self.agent_one)

                # AGENT TWO: getting the action and its probability from
                # predict function
                action_agent_two, action_probability_two = self.predict(
                    initial_state, self.agent_two)

                # Combine the actions of both agents to prepare to send to
                # environment class.
                actions = [action_agent_one, action_agent_two]

                ### PLAY THE GAME ###

                # Get the state, reward of both agents (as a list) and step
                # from environment class:
                state, reward_both_agents, round_counter = self.env.step(
                    actions)

                # Separate the rewards for each agent:
                reward_agent_one = reward_both_agents[0]
                reward_agent_two = reward_both_agents[1]

                ### Update the buffer for AGENT ONE ###
                # add the current state in a buffer
                batch_states_one.append(state)
                # add agent one's action in a buffer
                batch_actions_one.append(int(action_agent_one))
                # add agent one's reward in a buffer
                cur_rewards_one.append(reward_agent_one)

                ### Update the buffer for AGENT TWO ###
                # add the current state in a buffer
                batch_states_two.append(state)
                # add agent two's action in a buffer
                batch_actions_two.append(int(action_agent_two))
                # add agent one's reward in a buffer
                cur_rewards_two.append(reward_agent_two)

                # update the state of the environment for next round.
                initial_state = state

            # This loop is for training the agent after many IPDs
            if (i != 0) and (i % self.episodes == 0):

                                            # For average cumulative reward graphs:
                # CUMULATIVE REWARDS PER EPISODE:
                # calculate sum of rewards of all rounds in an episode for each
                # agent
                agent_one_sum_cur_rew = sum(cur_rewards_one)
                agent_two_sum_cum_rew = sum(cur_rewards_two)

                # AVERAGE CUMULATIVE REWARD
                # divide the cumulative reward per episode by total number of
                # rounds for each agent
                agent_one_avg_rew = agent_one_sum_cur_rew / round_counter
                agent_two_avg_rew = agent_two_sum_cum_rew / round_counter

                agent_one_avg_cum_rew.append(agent_one_avg_rew)
                agent_two_avg_cum_rew.append(agent_two_avg_rew)

                ### DISCOUNTED REWARD CALCULATION ###
                # Discounted reward calculation for agent one and addition of
                # that information to a buffer.
                batch_qvals_agent_one.extend(
                    self.discounted_rewards(
                        (cur_rewards_one), GAMMA=self.GAMMA))
                # Discounted reward calculation for agent two and addition of
                # that information to a buffer.
                batch_qvals_agent_two.extend(
                    self.discounted_rewards(
                        (cur_rewards_two), GAMMA=self.GAMMA))

                ### CONVERSION OF BUFFER LISTS INTO TENSORS ###
                # Convert states, actions and discounted_rewards buffers from
                # lists to tensors for agent one:
                batch_states_t_agent_one, batch_actions_t_agent_one, batch_qvals_t_agent_one = self.tensor_conversion(
                    batch_states_one, batch_actions_one, batch_qvals_agent_one)
                # Convert states, actions and discounted_rewards buffers from
                # lists to tensors for agent one:
                batch_states_t_agent_two, batch_actions_t_agent_two, batch_qvals_t_agent_two = self.tensor_conversion(
                    batch_states_two, batch_actions_two, batch_qvals_agent_two)

                ### TRAINING ###

                # AGENT ONE
                # Call the train function to train agent one and get the loss
                # tensor
                loss_agent_one = self.train(
                    self.agent_one,
                    self.optimizer_agent_one,
                    batch_states_t_agent_one,
                    batch_qvals_t_agent_one,
                    batch_actions_t_agent_one)

                # extracting the loss value from tensor
                loss_value_agent_one = loss_agent_one.item()

                # appending loss value to the list
                agent_one_loss_history.append(loss_value_agent_one)

                # PROBABILITY OF COOPERATION DEPENDING ON STATE

                # First Ever Round:
                agent_one_prob_first = self.get_single_state_strategy(
                    agent=self.agent_one, state="o")
                agent_one_first.append(agent_one_prob_first)
                # R
                agent_one_prob_r = self.get_single_state_strategy(
                    agent=self.agent_one, state="r")
                agent_one_r.append(agent_one_prob_r)
                # S
                agent_one_prob_s = self.get_single_state_strategy(
                    agent=self.agent_one, state="s")
                agent_one_s.append(agent_one_prob_s)
                # T
                agent_one_prob_t = self.get_single_state_strategy(
                    agent=self.agent_one, state="t")
                agent_one_t.append(agent_one_prob_t)
                # P
                agent_one_prob_p = self.get_single_state_strategy(
                    agent=self.agent_one, state="p")
                agent_one_p.append(agent_one_prob_p)

                # AGENT TWO
                # Call the train function to train agent one and get the loss
                # tensor
                loss_agent_two = self.train(
                    self.agent_two,
                    self.optimizer_agent_two,
                    batch_states_t_agent_two,
                    batch_qvals_t_agent_two,
                    batch_actions_t_agent_two)

                # extracting the loss value from tensor
                loss_value_agent_two = loss_agent_two.item()

                # appending loss value to the list
                agent_two_loss_history.append(loss_value_agent_two)

                # PROBABILITY OF COOPERATION DEPENDING ON STATE
                agent_two_prob_first = self.get_single_state_strategy(
                    agent=self.agent_two, state="o")
                agent_two_first.append(agent_two_prob_first)
                # R
                agent_two_prob_r = self.get_single_state_strategy(
                    agent=self.agent_two, state="r")
                agent_two_r.append(agent_two_prob_r)
                # S
                agent_two_prob_s = self.get_single_state_strategy(
                    agent=self.agent_two, state="s")
                agent_two_s.append(agent_two_prob_s)
                # T
                agent_two_prob_t = self.get_single_state_strategy(
                    agent=self.agent_two, state="t")
                agent_two_t.append(agent_two_prob_t)
                # P
                agent_two_prob_p = self.get_single_state_strategy(
                    agent=self.agent_two, state="p")
                agent_two_p.append(agent_two_prob_p)

            # An index of episodes:
                episode_count.append(i)

            i = + 1

        self.output_as_excel(
            episode_count,
            agent_one_avg_cum_rew,
            agent_two_avg_cum_rew,
            agent_one_loss_history,
            agent_two_loss_history,
            agent_one_first,
            agent_two_first,
            agent_one_r,
            agent_two_r,
            agent_one_s,
            agent_two_s,
            agent_one_t,
            agent_two_t,
            agent_one_p,
            agent_two_p,
            self.file_name,
            sheet_name="Agent_One_and_Two_Information")

    def output_as_excel(self, episode_index,
                        avg_cum_reward_one, avg_cum_reward_two,
                        loss_value_one, loss_value_two,
                        agent_one_first, agent_two_first,
                        agent_one_r, agent_two_r,
                        agent_one_s, agent_two_s,
                        agent_one_t, agent_two_t,
                        agent_one_p, agent_two_p,
                        file_name, sheet_name):

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

        df_agent = pd.DataFrame({
            'Episode': episode_index,
            'Agent_One_Avg_Cumulative_Reward': avg_cum_reward_one,
            'Agent_Two_Avg_Cumulative_Reward': avg_cum_reward_two,
            'Agent_One_Loss_Value': loss_value_one,
            'Agent_Two_Loss_Value': loss_value_two,
            'Agent_One_Prob_First_Round': agent_one_first,
            'Agent_Two_Prob_First_Round': agent_two_first,
            'Agent_One_Prob_R': agent_one_r,
            'Agent_Two_Prob_R': agent_two_r,
            'Agent_One_Prob_S': agent_one_s,
            'Agent_Two_Prob_S': agent_two_s,
            'Agent_One_Prob_T': agent_one_t,
            'Agent_Two_Prob_T': agent_two_t,
            'Agent_One_Prob_P': agent_one_p,
            'Agent_Two_Prob_P': agent_two_p,
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
            df_agent.to_excel(writer, sheet_name=sheet_name)
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

        # Use the np.random.choice and argmax to get the predicted action for
        # agent one
        action = np.random.choice(
            action_probability_array[0],
            p=action_probability_array[0])
        action = np.argmax(action_probability_array == action)

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
    conf["Seed"] = 45547611
    conf["Output_filename"] = "Two_player_lr0.001_100ep_100000runs.xlsx"
    conf["Episodes"] = 100
    conf["Runs"] = 1000000
    print(json.dumps(conf, indent=4))


def print_usage():
    print("Example 1: python Two_Agent_Settings.py ")
    print("will print an example json file")
    print()
    print("Example 2: python Two_Agent_Settings.py example.json")
    print("will run the simulation with the parameters in example.json")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_example_json()
    elif len(sys.argv) == 2:
        parameter_dictionary = json.loads(open(sys.argv[1]).read())
        game = Play(parameter_dictionary)
        game.gaming()