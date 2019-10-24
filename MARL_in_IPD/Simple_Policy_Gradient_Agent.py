import torch.nn as nn
import torch

# The Class contains a simple policy gradient agent (a neural network).
# It accepts size of input (in this case, number of states), number of actions and seed as arguments.
# For our Iterated Prisoner's Dilemma game, input_size is 5 and n_actions is 2.

class Simple_Policy_Gradient(nn.Module):
    def __init__(self, input_size, n_actions,my_seed):
        super(Simple_Policy_Gradient, self).__init__()

        torch.manual_seed(my_seed)

        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)          # dimension of softmax must be -1 to calculate softmax correctly across rows.
        )

    def forward(self, x):
        return self.net(x)