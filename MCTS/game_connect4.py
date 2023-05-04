import random
import sys

import torch
import numpy as np
import pygame

from MCTS.AI import MCTSfindMove, loadModel
from MCTS.gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from MCTS.interface import (draw, gameOver, initializeGame,
                       resolveEvent)

# Configurations
SIMULATIONS = 1000
WIDTH = 120
HEIGHT = WIDTH*0.8
UCB1 = 1.4

FILE = '/home/anton/skola/egen/pytorch/connect4/models/Connect4model10V1.pth'


def main() -> None:
    model1, device = loadModel(FILE)
    model2 = []
    for i in range(10):
        model2.append(
            loadModel(f'./connect4/models/Connect4model10V1-{i+1}.pth')[0])

    result = [0, 0, 0]
    # with torch.no_grad():
    #     for i in range(100):
    #         res = validationGame(model1, model2, device)
    #         result[res] += 1
    #         print(result)
    #     print('Wins player 1, Draws, Wins player -1')

    gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
    row, col = game(gameState, screen, frame, model2, device)
    if not gameOver(screen, gameEnd(gameState, row, col), WIDTH):
        main()


def game(gameState: np.ndarray, screen: pygame.Surface, frame: pygame.Surface, model: torch.nn.Module, device: torch.device) -> tuple:
    player = random.choice([1, -1])
    draw(screen, frame, gameState, WIDTH, HEIGHT)
    while True:
        if player == 1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            row = makeMove(gameState, player, move)
            if type(move) == int:
                player = nextPlayer(player)

        elif player == -1:
            # Neural network
            # print('Prediction after human move:', torch.softmax(model(model.board2tensor(
            #    gameState, player, device))[0][0], dim=0))
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model, device, cutoff=True)
            row = makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)
            # print(f'Prediction after AIs move:', torch.softmax(model(model.board2tensor(
            #    gameState, player, device))[0][0], dim=0))

        draw(screen, frame, gameState, WIDTH, HEIGHT, move, player)
        if gameEnd(gameState, row, move).any():
            return (row, move)

        if not availableMoves(gameState):
            return (row, move)


def validationGame(model1: torch.nn.Module, model2: torch.nn.Module, device: torch.device) -> int:
    player = random.choice([1, -1])
    gameState = np.zeros((6, 7))
    while True:
        if player == 1:
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model=model1, device=device, cutoff=True)

        elif player == -1:
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model=model2, device=device, cutoff=True)

        row = makeMove(gameState, player, move)
        player = nextPlayer(player)
        resolveEvent(gameState, 0, WIDTH)

        if gameEnd(gameState, row, move).any():
            return 0 if player == -1 else 2
        elif not availableMoves(gameState):
            return 1


if __name__ == '__main__':
    with torch.no_grad():
        main()
