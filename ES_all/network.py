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
    state_dim = 11
    action_dim = 5

    model = FeedForwardNN(state_dim, action_dim)

    fc1_weights = model.fc1.weight.data
    fc1_bias = model.fc1.bias.data
    fc2_weights = model.fc2.weight.data
    fc2_bias = model.fc2.bias.data
    fc3_weights = model.fc3.weight.data
    fc3_bias = model.fc3.bias.data
    fc4_weights = model.fc4.weight.data
    fc4_bias = model.fc4.bias.data

    print(f"fc1_weights shape: {fc1_weights.shape}, fc1_bias shape: {fc1_bias.shape}")
    print(f"fc2_weights shape: {fc2_weights.shape}, fc2_bias shape: {fc2_bias.shape}")
    print(f"fc3_weights shape: {fc3_weights.shape}, fc3_bias shape: {fc3_bias.shape}")
    print(f"fc4_weights shape: {fc4_weights.shape}, fc4_bias shape: {fc4_bias.shape}")
    print()

    sigma = 0.1
    noise = torch.randn(5) * sigma

    print(f"noise: {noise}")

    theta = torch.nn.utils.parameters_to_vector(model.parameters())
    print(f"theta shape: {theta.shape}")
    print()

    noise_theta = theta + noise[0]

    torch.nn.utils.vector_to_parameters(noise_theta, model.parameters())

    fc1_weights_new = model.fc1.weight.data
    fc1_bias_new = model.fc1.bias.data
    fc2_weights_new = model.fc2.weight.data
    fc2_bias_new = model.fc2.bias.data
    fc3_weights_new = model.fc3.weight.data
    fc3_bias_new = model.fc3.bias.data
    fc4_weights_new = model.fc4.weight.data
    fc4_bias_new = model.fc4.bias.data

    print(f"fc1_weights_new shape: {fc1_weights_new.shape}, fc1_bias_new shape: {fc1_bias_new.shape}")
    print(f"fc2_weights_new shape: {fc2_weights_new.shape}, fc2_bias_new shape: {fc2_bias_new.shape}")
    print(f"fc3_weights_new shape: {fc3_weights_new.shape}, fc3_bias_new shape: {fc3_bias_new.shape}")
    print(f"fc4_weights_new shape: {fc4_weights_new.shape}, fc4_bias_new shape: {fc4_bias_new.shape}")
    print()

    fc1_equal = torch.allclose(fc1_weights, fc1_weights_new)
    fc2_equal = torch.allclose(fc2_weights, fc2_weights_new)
    fc3_equal = torch.allclose(fc3_weights, fc3_weights_new)
    fc4_equal = torch.allclose(fc4_weights, fc4_weights_new)

    fc1_difference = fc1_weights_new - fc1_weights
    print(f"fc1_difference: {fc1_difference}")

    print(f"fc1_equal: {fc1_equal}, fc2_equal: {fc2_equal}, fc3_equal: {fc3_equal}, fc4_equal: {fc4_equal}")
    print()

    fc1_bias_equal = torch.allclose(fc1_bias, fc1_bias_new)
    fc2_bias_equal = torch.allclose(fc2_bias, fc2_bias_new)
    fc3_bias_equal = torch.allclose(fc3_bias, fc3_bias_new)
    fc4_bias_equal = torch.allclose(fc4_bias, fc4_bias_new)

    print(f"fc1_bias_equal: {fc1_bias_equal}, fc2_bias_equal: {fc2_bias_equal}, fc3_bias_equal: {fc3_bias_equal}, fc4_bias_equal: {fc4_bias_equal}")