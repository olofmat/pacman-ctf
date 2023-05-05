import numpy as np
import random
from capture import GameState


def randomMove(moves: list) -> int:
    return random.choice(moves)


def nextPlayer(player: int) -> int:
    return (player+1) % 4


def result(gameState:GameState):
    if gameState.getScore() > 0: return np.array([1, 0])
    return np.array([0, 1])