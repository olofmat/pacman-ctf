import time

import numpy as np

from capture import GameState
from MCTS.Node import Node
from MCTS.MCTSData import MCTSData


def MCTSfindMove(data:MCTSData) -> str:
    moves = data.state.getLegalActions(data.player)
    removeStop(moves)
    if not moves:
        return None
    
    starting_position = data.state.getAgentPosition(data.player)
    furthest_away_distance = 0
    furthest_away_position = starting_position
    max_depth = 0

    players = []
    for i in range(4):
        pos = data.state.getAgentPosition(i)
        if not pos: continue
        # if data.player == i or (((pos[0]-starting_position[0])**2 + (pos[1]-starting_position[1])**2 <= 25) and (data.state.isOnRedTeam(i) != data.state.isOnRedTeam(data.player))): players.append(i)
        if data.player == i or not (data.state.isOnRedTeam(i) == data.state.isOnRedTeam(data.player)) or data.distances[pos[0]][pos[1]][starting_position[0]][starting_position[1]] <= 3: players.append(i)
        # if data.player == i or not (data.state.isOnRedTeam(i) == data.state.isOnRedTeam(data.player)) or (pos[0]-starting_position[0])**2 + (pos[1]-starting_position[1])**2 <= 9: players.append(i)
    players = np.array(players)
    
    if data.saw_last_round == False and len(players) == 1:
        data.root = data.root.chooseBestChild()
        print(f"player {data.player} is keeping tree")
    else:
        data.root = Node(data.player)
        data.root.makeChildren(data.player, moves)
    
    data.saw_last_round = False if len(players) == 1 else True

    startTime = time.process_time()
    for _ in range(data.sim_number):
        if time.process_time() - startTime > data.sim_time:
            distance = [x-y for x, y in zip(furthest_away_position, starting_position)]
            print(f"maximum depth: {max_depth}, checked {distance} from starting position in {starting_position}")
            printData(data.root)
            return data.root.chooseBestChild().move

        currentState = GameState(data.state)
        current = data.root
        current_depth = 0
        
        # Tree traverse
        while len(current.children) > 0:
            current_depth += 1
            current = current.selectChild(data.UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*data.sim_number:
                printData(data.root)
                return current.move

        if (current_depth > max_depth): max_depth = current_depth

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not currentState.isOver():
            moves = currentState.getLegalActions(current.nextPlayer(players))
            removeStop(moves)
            current.makeChildren(current.nextPlayer(players), moves)
            current = current.selectChild(data.UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)
            
            if (current.player == data.player):
                furthest_away_distance, furthest_away_position = calculate_depth(
                    currentState, starting_position, furthest_away_distance, furthest_away_position, data.player)

        # Rollout
        result = evaluationHeuristic(currentState)

        # Backpropagate
        current.backpropagate(data.state, result)

    printData(data.root)
    return data.root.chooseBestChild().move


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



def printData(root: Node) -> None:
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
