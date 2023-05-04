import numpy as np
import torch
import torch.nn as nn
import time

from Connect4Model import Model
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer, randomMove
from Node import Node


def MCTSfindMove(rootState: np.ndarray, rootPlayer: int, simulations: int, UCB1: float, model: nn.Module = None, device: torch.device = None, cutoff: bool = False) -> int:
    moves = availableMoves(rootState)

    if not moves:
        return None

    root = Node(rootPlayer)
    root.makeChildren(rootPlayer, moves)

    startTime = time.process_time()
    for _ in range(simulations):
        if time.process_time() - startTime > 10000:
            printData(root)
            return root.chooseMove()

        currentState = rootState.copy()
        current = root

        # Tree traverse
        while len(current.children) > 0:
            current = current.selectChild(UCB1)
            row = makeMove(currentState, current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*simulations:
                # printData(root)
                return current.move

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not gameEnd(currentState, row, current.move).any():
            moves = availableMoves(currentState)
            current.makeChildren(current.nextPlayer(), moves)
            current = current.selectChild(UCB1)
            row = makeMove(currentState, current.player, current.move)

        # Rollout
        result = rollout(currentState, current.nextPlayer(),
                         row, current.move, model, device, cutoff)

        # Backpropagate
        current.backpropagate(result)

    # printData(root)
    return root.chooseMove()


def rollout(currentState: np.ndarray, currentPlayer: int, row: int, move: int, model: nn.Module = None, device: torch.device = None, cutoff: bool = False) -> np.ndarray:
    # finds a random move and executes it if possible
    while True:
        result = gameEnd(currentState, row, move)
        if result.any():
            return result

        if cutoff:
            return evaluation(currentState, currentPlayer, model, device)

        moves = availableMoves(currentState)
        if not moves:
            return np.array([0, 0])

        move = randomMove(moves)
        row = makeMove(currentState, currentPlayer, move)
        currentPlayer = nextPlayer(currentPlayer)


def evaluation(board: np.ndarray, currentPlayer: int, model: nn.Module, device: torch.device) -> np.ndarray:
    returnValue = 0
    if type(model) == list:
        for m in model:
            input = m.board2tensor(board, currentPlayer, device)
            prob = m(input)[0][0]
            prob = torch.softmax(prob, dim=0)
            returnValue += (prob[0]-prob[2]).item()
        returnValue /= len(model)
        return np.array([returnValue, -returnValue])

    else:
        input = model.board2tensor(board, currentPlayer, device)
        prob = model(input)[0][0]
        prob = torch.softmax(prob, dim=0)
        eval = (prob[0]-prob[2]).item()
        return np.array([eval, -eval])


def loadModel(file: str = '/home/anton/skola/egen/pytorch/connect4/models/Connect4model200k.pth'):
    OUT_CHANNELS1, OUT_CHANNELS2, HIDDEN_SIZE1, HIDDEN_SIZE2 = 6, 6, 120, 72
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = Model(OUT_CHANNELS1, OUT_CHANNELS2,
                  HIDDEN_SIZE1, HIDDEN_SIZE2).to(device)
    model.load_state_dict(torch.load(file, map_location='cpu'))
    model.to(device)
    model.eval()

    return model, device


def printData(root: object) -> None:
    visits, val, p = root.visits, root.value, root.player
    print(
        f'root; player: {p}, rollouts: {visits}, value: {round(val*1, 2)}, vinstprocent: {round((visits+val)*50/visits, 2)}%')
    print('children;')
    print('visits:', end=' ')
    childVisits = [child.visits.tolist() for child in root.children]
    childVisits.sort(reverse=True)
    print(childVisits)
    print('')
