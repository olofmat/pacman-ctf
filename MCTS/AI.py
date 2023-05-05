import time

import numpy as np
import torch
import torch.nn as nn

from capture import GameState
from MCTS.Connect4Model import Model
from MCTS.gameplay import nextPlayer, randomMove, result
from MCTS.Node import Node


def MCTSfindMove(rootState: GameState, rootPlayer: int, UCB1: float, sim_time: float = np.inf, sim_number: int = 100_000_000, cutoff: int = 0, model: nn.Module = None, device: torch.device = None) -> str:
    moves = rootState.getLegalActions(rootPlayer)
    moves.remove('Stop')

    if not moves:
        return None

    root = Node(rootPlayer)
    root.makeChildren(rootPlayer, moves)

    startTime = time.process_time()
    for _ in range(sim_number):
        if time.process_time() - startTime > sim_time:
            printData(root)
            return root.chooseMove()

        currentState = GameState(rootState)
        current = root

        # Tree traverse
        while len(current.children) > 0:
            current = current.selectChild(UCB1)
            if currentState.getAgentPosition(current.player):
                currentState = currentState.generateSuccessor(current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*sim_number:
                # printData(root)
                return current.move

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not currentState.isOver():
            if currentState.getAgentPosition(current.nextPlayer()):
                moves = currentState.getLegalActions(current.nextPlayer())
                moves.remove('Stop')
                current.makeChildren(current.nextPlayer(), moves)
                current = current.selectChild(UCB1)
                currentState = currentState.generateSuccessor(current.player, current.move)
            else:
                moves = ['Stop']
                current.makeChildren(current.nextPlayer(), moves)
                current = current.selectChild(UCB1)

        # Rollout
        result = rolloutHeuristic(currentState, current.nextPlayer(), cutoff)

        # Backpropagate
        current.backpropagate(rootState, result)

    # printData(root)
    return root.chooseMove()

    
def rolloutHeuristic(currentState: GameState, currentPlayer: int, cutoff: int) -> np.ndarray:
    movesInRollout = 0
    # finds a random move and executes it if possible. as long as gameEnd is False and movesInRollout is less than cutoff
    while True:
        if currentState.isOver():
            return result()

        if movesInRollout >= cutoff:
            return evaluationHeuristic(currentState)

        if currentState.getAgentPosition(currentPlayer):
            moves = currentState.getLegalActions(currentPlayer)
            moves.remove('Stop')
        else:
            moves = ['Stop']
        move = randomMove(moves)
        currentState = currentState.generateSuccessor(currentPlayer, move)

        movesInRollout += 1
        currentPlayer = nextPlayer(currentPlayer)


def evaluationHeuristic(gameState: GameState) -> np.ndarray:
    foodCapturedByYou = gameState.data.layout.totalFood/2 - len(gameState.getBlueFood().asList())
    foodCapturedByOpponent = gameState.data.layout.totalFood/2 - len(gameState.getRedFood().asList())    
    score = gameState.getScore()

    ### RESONABLE HEURISTIC. Maximize your score. Maximize how much you're carrying but less so than how much you deposited.
    ### Minimize how much food your opponent has captured but it's harder so dont spend to much time on it.
    heuristic = score + 1/4*foodCapturedByYou - 1/4*foodCapturedByOpponent
    # print(f"my captured {foodCapturedByYou}, opponent captured: {foodCapturedByOpponent}, score: {score}")
    
    # heuristic = np.tanh(heuristic)

    return np.array([heuristic, -heuristic])



def loadModel(file: str = '/home/anton/skola/egen/pytorch/connect4/models/Connect4model200k.pth'):
    OUT_CHANNELS1, OUT_CHANNELS2, HIDDEN_SIZE1, HIDDEN_SIZE2 = 6, 6, 120, 72
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
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
    


# Hoppa över motståndarens stop om man inte ser dem 



# def rolloutNN(currentState: np.ndarray, currentPlayer: int, row: int, move: int, model: nn.Module = None, device: torch.device = None, cutoff: bool = False) -> np.ndarray:
#     # finds a random move and executes it if possible
#     while True:
#         result = gameEnd(currentState, row, move)
#         if result.any():
#             return result

#         if cutoff:
#             return evaluationNN(currentState, currentPlayer, model, device)

#         moves = availableMoves(currentState)
#         if not moves:
#             return np.array([0, 0])

#         move = randomMove(moves)
#         row = makeMove(currentState, currentPlayer, move)
#         currentPlayer = nextPlayer(currentPlayer)


# def evaluationNN(board: np.ndarray, currentPlayer: int, model: nn.Module, device: torch.device) -> np.ndarray:
#     returnValue = 0
#     if type(model) == list:
#         for m in model:
#             input = m.board2tensor(board, currentPlayer, device)
#             prob = m(input)[0][0]
#             prob = torch.softmax(prob, dim=0)
#             returnValue += (prob[0]-prob[2]).item()
#         returnValue /= len(model)
#         return np.array([returnValue, -returnValue])

#     else:
#         input = model.board2tensor(board, currentPlayer, device)
#         prob = model(input)[0][0]
#         prob = torch.softmax(prob, dim=0)
#         eval = (prob[0]-prob[2]).item()
#         return np.array([eval, -eval])