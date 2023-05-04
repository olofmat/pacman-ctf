import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """Softmax of output is probability of winning or drawing as 
    [win(1), draw, win(-1)]. Input to forward is a (3, 6, 7) tensor where 
    first layer represents player 1, second layer represents player -1, 
    and third layer is filled with the value of player to make a move."""

    def __init__(self, out_channels1: int, out_channels2: int, hidden_dim1: int, hidden_dim2: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, out_channels1, 2)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, 2)
        self.linear1 = nn.Linear(20*out_channels2, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 3)

    def forward(self, x) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.linear1(x.reshape(-1, 1, 120)))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def board2tensor(self, board: np.ndarray, player: int, device: torch.device) -> torch.Tensor:
        board = torch.from_numpy(board).to(device)
        ones = torch.ones((6, 7)).to(device)
        a = (board == ones).float()
        b = (board == -ones).float()

        input = torch.empty((1, 3, 6, 7), device=device)
        input[0][0] = a
        input[0][1] = b
        input[0][2] = player*ones
        return input
