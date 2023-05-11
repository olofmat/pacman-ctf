import time

import numpy as np

from capture import GameState
from MCTS.Node import Node


def MCTSfindMove(rootState: GameState, rootPlayer: int, UCB1: float, sim_time: float = np.inf, sim_number: int = 100000000) -> str:
    moves = rootState.getLegalActions(rootPlayer)
    removeStop(moves)
    if not moves:
        return None
    
    starting_position = rootState.getAgentPosition(rootPlayer)
    furthest_away_distance = 0
    furthest_away_position = starting_position
    max_depth = 0

    players = []
    for i in range(4):
        pos = rootState.getAgentPosition(i)
        if not pos: continue
        if rootPlayer == i or not (rootState.isOnRedTeam(i) == rootState.isOnRedTeam(rootPlayer)) or (pos[0]-starting_position[0])**2 + (pos[1]-starting_position[1])**2 <= 25: players.append(i)
    players = np.array(players)

    root = Node(rootPlayer)
    root.makeChildren(rootPlayer, moves)

    startTime = time.process_time()
    for _ in range(sim_number):
        if time.process_time() - startTime > sim_time:
            distance = [x-y for x, y in zip(furthest_away_position, starting_position)]
            print(f"maximum depth: {max_depth}, checked {distance} from starting position in {starting_position}")
            printData(root)
            return root.chooseMove()

        currentState = GameState(rootState)
        current = root
        current_depth = 0
        
        # Tree traverse
        while len(current.children) > 0:
            current_depth += 1
            current = current.selectChild(UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*sim_number:
                printData(root)
                return current.move

        if (current_depth > max_depth): max_depth = current_depth

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not currentState.isOver():
            moves = currentState.getLegalActions(current.nextPlayer(players))
            removeStop(moves)
            current.makeChildren(current.nextPlayer(players), moves)
            current = current.selectChild(UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)
            
            if (current.player == rootPlayer):
                furthest_away_distance, furthest_away_position = calculate_depth(
                    currentState, starting_position, furthest_away_distance, furthest_away_position, rootPlayer)

        # Rollout
        result = evaluationHeuristic(currentState)

        # Backpropagate
        current.backpropagate(rootState, result)

    printData(root)
    return root.chooseMove()


def evaluationHeuristic(gameState: GameState) -> float:
    foodCapturedByYou = gameState.data.layout.totalFood/2 - len(gameState.getBlueFood().asList())
    foodCapturedByOpponent = gameState.data.layout.totalFood/2 - len(gameState.getRedFood().asList())    
    score = gameState.getScore()

    ### REASONABLE HEURISTIC. Maximize your score. Maximize how much you're carrying but less so than how much you deposited.
    ### Minimize how much food your opponent has captured but it's harder so dont spend to much time on it.
    heuristic = score + 1/4*foodCapturedByYou - 1/4*foodCapturedByOpponent
    # print(f"my captured {foodCapturedByYou}, opponent captured: {foodCapturedByOpponent}, score: {score}")
    
    # heuristic = np.tanh(heuristic)
    return heuristic


def removeStop(list:list) -> None:
    try:
        list.remove('Stop')
    except ValueError:
        pass
    
    
def calculate_depth(gameState:GameState, starting_position, furthest_away_distance:int, furthest_away_position:int, rootPlayer:int):
    new_pos = gameState.getAgentPosition(rootPlayer)
    difference = (new_pos[0]-starting_position[0],new_pos[1]-starting_position[1])
    dist = np.sqrt(difference[0]**2+difference[1]**2)
    if(dist>furthest_away_distance):
        furthest_away_position = gameState.getAgentPosition(rootPlayer)
        furthest_away_distance = dist
    
    return furthest_away_distance, furthest_away_position



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
    
    
# def rolloutHeuristic(currentState: GameState, currentPlayer: int, cutoff: int) -> float:
#     movesInRollout = 0
#     # finds a random move and executes it if possible. as long as gameEnd is False and movesInRollout is less than cutoff
#     while True:
#         if currentState.isOver():
#             return result()

#         if movesInRollout >= cutoff:
#             return evaluationHeuristic(currentState)

#         moves = currentState.getLegalActions(currentPlayer)
#         removeStop(moves)
#         move = randomMove(moves)
#         currentState = currentState.generateSuccessor(currentPlayer, move)

#         movesInRollout += 1
#         currentPlayer = nextPlayer(currentPlayer)
