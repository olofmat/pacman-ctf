import numpy as np
import random
from capture import GameState


def randomMove(moves: list) -> int:
    return random.choice(moves)


def nextPlayer(player: int, players:np.ndarray) -> int:
    player_index = np.where(players == player)
    return (player_index[0][0]+1) % players.size


def result(gameState:GameState) -> float:
    if gameState.getScore() > 0: 1
    return -1