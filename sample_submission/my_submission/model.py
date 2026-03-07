import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

class ResBlock(nn.Module):
    """A lightweight residual block for fast CPU execution."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class Connect4CNN(nn.Module):
    """
    Lightweight ResNet architecture optimized for Connect 4 Plus.
    Input: (Batch, 3, 6, 7) - My pieces, Opponent pieces, Neutral Coin
    Output: (Batch, 7) - Q-values or logits for each column
    """
    def __init__(self):
        super().__init__()
        # Input convolution: 3 channels in, 64 out
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(64)
        
        # 3 Residual blocks allows sufficient receptive field without being too heavy
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        
        # Policy/Q-Value Head
        self.conv_head = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(2)
        self.fc_head = nn.Linear(2 * 6 * 7, 7)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = F.relu(self.bn_head(self.conv_head(x)))
        x = x.view(x.size(0), -1)
        return self.fc_head(x)

class Bot:
    def __init__(self):
        # Force CPU to comply strictly with the rules and avoid CUDA overhead
        self.device = torch.device('cpu')
        
        # Initialize architecture
        self.model = Connect4CNN()
        self.model.eval()
        
        # Load weights
        weights_path = os.path.join(os.path.dirname(__file__), 'model.safetensors')
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
            
        # Optional PyTorch optimization for single-thread environments
        # Restricts OpenMP thread overhead which can be detrimental for tiny models
        torch.set_num_threads(1)

    def act(self, observation):
        # Board shape: (6, 7, 3), action_mask shape: (7,)
        board = observation["observation"]
        action_mask = observation["action_mask"]
        
        # Prepare input tensor: shape (1, 3, 6, 7)
        board_tensor = torch.from_numpy(board).float()
        # Permute (H, W, Channels) -> (Channels, H, W)
        board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get logits/Q-values for all 7 columns
            logits = self.model(board_tensor).squeeze(0)
            
        # Mask out illegal actions with -inf
        for i in range(7):
            if action_mask[i] == 0:
                logits[i] = float('-inf')
                
        # Return the column with the highest value
        return int(torch.argmax(logits).item())
