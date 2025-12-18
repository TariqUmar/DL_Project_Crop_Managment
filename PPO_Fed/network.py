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


    torch.manual_seed(123)

    net1 = FeedForwardNN(25, 25) # actor
    net2 = FeedForwardNN(25, 1) # critic
    theta1 = net1.parameters_to_vector().clone()
    theta2 = net2.parameters_to_vector().clone()

    print("FOR FEDERATED PPO")
    print("--------------------------------")
    print(theta1.type())
    print(theta2.type())
    # The size of the theta vector in MB
    print(theta1.shape[0]*theta1.element_size(), "theta1 Bytes")
    print(theta2.shape[0]*theta2.element_size(), "theta2 Bytes")
    print(theta1.shape[0]*theta1.element_size()/1024, "theta1 KB")
    print(theta2.shape[0]*theta2.element_size()/1024, "theta2 KB")
    MB1 = theta1.shape[0]*theta1.element_size()/1024/1024
    MB2 = theta2.shape[0]*theta2.element_size()/1024/1024
    print(theta1.element_size(), "theta1 element size")
    print(theta2.element_size(), "theta2 element size")
    print(MB1, "theta1 MB")
    print(MB2, "theta2 MB")
    print(MB1*5, "theta1 MB for 5 clients")
    print(MB2*5, "theta2 MB for 5 clients")

    MB = MB1 + MB2

    print(MB, "MB for 5 clients")

    UPLINK_BW = 0.125 # 1 Mbps = 0.125 MBps
    DOWNLINK_BW = 1.25 # 10 Mbps = 1.25 MBps

    uplink_time = MB*5/UPLINK_BW
    downlink_time = MB*5/DOWNLINK_BW
    print(uplink_time, "seconds for uplink")
    print(downlink_time, "seconds for downlink")

    total_time = uplink_time + downlink_time
    print(total_time, "seconds for total time")

    print("--------------------------------")

    print("FOR Evolutionary Algos")
    print("--------------------------------")
    noise = torch.randn(128, dtype=torch.float32)
    print(noise.shape[0]*noise.element_size(), "Bytes")
    print(noise.shape[0]*noise.element_size()/1024, "KB")
    MB = noise.shape[0]*noise.element_size()/1024/1024
    print(noise.element_size(), "noise element size")
    print(MB, "MB")
    print(MB*5, "MB for 100 clients")

    uplink_time = MB*5/UPLINK_BW
    downlink_time = MB*5/DOWNLINK_BW
    print(uplink_time, "seconds for uplink")
    print(downlink_time, "seconds for downlink")

    total_time = uplink_time + downlink_time
    print(total_time, "seconds for total time")
    print("--------------------------------")