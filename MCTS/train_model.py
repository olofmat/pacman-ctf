import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from AI import MCTSfindMove
from Connect4Model import Model
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer


# Saving/Loading
LOAD_MODEL = False
SAVE_MODEL = True
LOAD_FILE = '/home/anton/skola/egen/pytorch/connect4/models/Connect4model100k.pth'
SAVE_FILE = '/home/anton/skola/egen/pytorch/connect4/models/Connect4model10V1.pth'
GAMES_FILE400 = '/home/anton/skola/egen/pytorch/connect4/training_games400.npy'
GAMES_FILE1 = '/home/anton/skola/egen/pytorch/connect4/training_games1.npy'

# Learning
LEARNING_RATE = 0.01
N_EPOCHS = 10
BATCH_SIZE = 16
N_GAMES = 1_000

# Model architecture
OUT_CHANNELS1 = 6
OUT_CHANNELS2 = 6
HIDDEN_SIZE1 = 120
HIDDEN_SIZE2 = 72

# MCTS
SIMULATIONS = 1
UCB1 = 1.4


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    for i in range(10):
        model = Model(OUT_CHANNELS1, OUT_CHANNELS2,
                      HIDDEN_SIZE1, HIDDEN_SIZE2).to(device)
        if LOAD_MODEL:
            model.load_state_dict(torch.load(LOAD_FILE, map_location='cpu'))
            model.to(device)
            model.eval()

        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        loss = nn.CrossEntropyLoss()

        dataset = GamesDataset()
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                 num_workers=2, shuffle=True)

        # save_games(GAMES_FILE1)

        train(model, data_loader, optimizer, loss)
        # validate(model, device)

        if SAVE_MODEL:
            torch.save(model.state_dict(),
                       f'./connect4/models/Connect4model10V1-{i+1}.pth')


class GamesDataset(Dataset):
    def __init__(self):
        # with open(GAMES_FILE1, 'rb') as f:
        #     xy1 = np.load(f)
        with open(GAMES_FILE400, 'rb') as f:
            xy400 = np.load(f)
        xy400 = xy400[:64000]

        self.n_samples = xy400.shape[0]
        self.data = torch.from_numpy(
            np.array([x[:3] for x in xy400])).to(torch.float32)
        self.labels = torch.from_numpy(
            np.array([x[-1][0][0] for x in xy400])).to(torch.int64)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.n_samples


def train(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, loss: nn.CrossEntropyLoss) -> None:
    for epoch in range(N_EPOCHS):
        for i, (data, label) in enumerate(data_loader):

            predictions = model(data)

            error = loss(predictions.reshape((BATCH_SIZE, 3)), label)

            error.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (i+1) % 1000 == 0:
                print(
                    f'Epoch [{epoch+1}/{N_EPOCHS}], Batch: {i+1}/{len(data_loader)}, Loss: {error.item():.4f}, '
                    f'prediction: {torch.softmax(predictions.reshape((BATCH_SIZE, 3))[0], dim=0)[label[0].item()].item():.4f}, '
                    f'result: {label[0].item()}')


def save_games(GAMES_FILE):
    all_games = training_game()

    for i in range(N_GAMES):
        new_game = training_game()
        all_games = np.concatenate((all_games, new_game), axis=0)

        if (i+1) % (N_GAMES/100) == 0:
            print(f'Game {i+1}/{N_GAMES}')

    with open(GAMES_FILE, 'wb') as file:
        np.save(file, all_games)


def training_game() -> np.ndarray:
    player = random.choice([1, -1])
    gameState = np.zeros((6, 7))
    game_history = np.zeros((1, 4, 6, 7))
    game_history[0][2] = np.ones((6, 7)) * player
    while True:
        move = MCTSfindMove(gameState, player, SIMULATIONS, UCB1)
        row = makeMove(gameState, player, move)
        player = nextPlayer(player)

        # adding flipped boards to history
        game_history = add_to_history(game_history, gameState, player)

        # win
        if gameEnd(gameState, row, move).any():
            # adding endboard but with other player to make turn
            game_history = add_to_history(game_history, gameState, -player)

            res = 0 if nextPlayer(player) == 1 else 2
            for i in range(game_history.shape[0]):
                game_history[i][-1] = res

            return game_history

        # draw
        if not availableMoves(gameState):
            # adding endboard but with other player to make turn
            game_history = add_to_history(game_history, gameState, -player)

            for i in range(game_history.shape[0]):
                game_history[i][-1] = 1

            return game_history


def add_to_history(game_history: np.ndarray, gameState: np.ndarray, player: int) -> np.ndarray:
    original = board2input(gameState, player)
    flipped = board2input(np.flip(gameState, 1).copy(), player)
    game_history = np.concatenate((game_history, original), axis=0)
    game_history = np.concatenate((game_history, flipped), axis=0)
    return game_history


def board2input(board: np.ndarray, player: int) -> np.ndarray:
    ones = np.ones((6, 7))
    a = (board == ones).astype(np.float32)
    b = (board == -ones).astype(np.float32)

    input = np.empty((1, 4, 6, 7))
    input[0][0] = a
    input[0][1] = b
    input[0][2] = player*ones
    return input


if __name__ == '__main__':
    main()
