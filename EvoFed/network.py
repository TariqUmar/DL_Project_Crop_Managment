import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):  # Initialize the FeedForwardNN with the given state and action dimensions
        '''
        Initialize the FeedForwardNN.
        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.

        Current Architecture:

            Input -> 50 -> 50 -> 50 -> action_dim

        '''

        # torch.manual_seed(seed)

        super(FeedForwardNN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def __str__(self): # We will see this output when we print an instance of the QNetwork class
        output = f"FeedForwardNN(state_dim={self.state_dim}, action_dim={self.action_dim})\n" + \
                 f"Current Architecture: state_dim -> 50 -> 50 -> 50 -> action_dim"
        return output

    def initialize_weights(self): # Initialize the weights of the FeedForwardNN
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        
        # Convert to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.fc1.weight.device)

        activation1 = F.relu(self.fc1(obs))
        activation2 = F.relu(self.fc2(activation1))
        activation3 = F.relu(self.fc3(activation2))
        output = self.fc4(activation3)
        return output

    def parameters_to_vector(self):
        """Flatten all model parameters into a single 1-D tensor (theta)."""
        # collect all parameter tensors, flatten each, and concatenate
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def vector_to_parameters(self, theta_vec):
        """Load parameters from a flat 1-D tensor back into the model."""
        # pointer keeps track of where we are in the flat vector
        pointer = 0
        for p in self.parameters():
            # number of elements in this parameter
            num_params = p.numel()
            # reshape the corresponding slice and copy it into p
            p.data.copy_(theta_vec[pointer:pointer + num_params].view_as(p))
            pointer += num_params


if __name__ == "__main__":


    torch.manual_seed(42)

    net1 = FeedForwardNN(25, 5)
    theta = net1.parameters_to_vector().clone()

    print(theta.type())

    torch.manual_seed(42)

    net2 = FeedForwardNN(25, 5)
    # net2.vector_to_parameters(theta)

    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        print(torch.allclose(p1, p2))