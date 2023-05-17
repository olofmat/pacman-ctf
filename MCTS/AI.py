import time

import numpy as np

from capture import GameState
from MCTS.Node import Node
from MCTS.MCTSData import MCTSData


def MCTSfindMove(data:MCTSData) -> str:
    moves = data.state.getLegalActions(data.player)
    removeStop(moves)
    if not moves: return None
    
    starting_position = data.state.getAgentPosition(data.player)
    furthest_away_distance = 0
    furthest_away_position = starting_position
    max_depth = 0
    
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
            moves = currentState.getLegalActions(current.nextPlayer(data.players))
            removeStop(moves)
            current.makeChildren(current.nextPlayer(data.players), moves)
            current = current.selectChild(data.UCB1)
            currentState = currentState.generateSuccessor(current.player, current.move)
            
            if (current.player == data.player):
                furthest_away_distance, furthest_away_position = calculate_depth(
                    currentState, starting_position, furthest_away_distance, furthest_away_position, data.player)

        # Rollout
        result = evaluationHeuristic(currentState, data, current.player)

        # Backpropagate
        current.backpropagate(data.state, result)

    printData(data.root)
    return data.root.chooseBestChild().move


def evaluationHeuristic(gameState: GameState, data:MCTSData, current_player:int) -> tuple[float]:
    ### REASONABLE HEURISTIC. Maximize your score. Maximize how much you're carrying but less so than how much you deposited.
    ### Minimize how much food your opponent has captured but it's harder so dont spend to much time on it.
    
    # carried and scored food
    foodCapturedByRed  = gameState.data.layout.totalFood/2 - len(gameState.getBlueFood().asList())
    foodCapturedByBlue = gameState.data.layout.totalFood/2 - len(gameState.getRedFood().asList())    
    score = gameState.getScore()
    
    # penalty if you are on home row
    home_penalty_red, home_penalty_blue = 0, 0
    for i in range(4):
        pos = gameState.getAgentPosition(i)
        if not pos: continue
        if gameState.isOnRedTeam(i)     and pos[0] == 1: home_penalty_red += 1
        if not gameState.isOnRedTeam(i) and pos[0] == gameState.data.layout.width-2: home_penalty_blue += 1
        
    # summing
    heuristic_red  =  score + foodCapturedByRed/4 - foodCapturedByBlue/4 - home_penalty_red + home_penalty_blue
    heuristic_blue = -score - foodCapturedByRed/4 + foodCapturedByBlue/4 + home_penalty_red - home_penalty_blue
    
    if gameState.isOnRedTeam(data.player) != gameState.isOnRedTeam(current_player): return heuristic_red, heuristic_blue
    
    # distance to closest food
    data.get_food_locations()
    my_pos = gameState.getAgentPosition(current_player)
    closest_food = 100
    for food_location in data.food:
        closest_food = min(closest_food, data.distances[my_pos[0]][my_pos[1]][food_location[0]][food_location[1]])
        
        
    middle = (gameState.data.layout.walls.width-1)/2
    closest_enemy = 100
    closest_capsule = 100
    for dist in data.distributions:
        for pos in dist:
            if (pos[0] > middle and gameState.isOnRedTeam(data.player)) or (pos[0] < middle and not gameState.isOnRedTeam(data.player)): continue
            closest_enemy = min(closest_enemy, data.distances[my_pos[0]][my_pos[1]][pos[0]][pos[1]])
            
    if closest_enemy == 100: 
        closest_enemy = 0 
    else:
        for capsule in gameState.getRedCapsules() if gameState.isOnRedTeam(data.player) else gameState.getBlueCapsules():
            closest_capsule = min(closest_capsule, data.distances[my_pos[0]][my_pos[1]][capsule[0]][capsule[1]])    
    if closest_capsule == 100: closest_capsule = 0 
            
    if gameState.isOnRedTeam(data.player):
        if data.player <= 1: heuristic_red += (1-closest_food/76)/8 - (1-closest_enemy/76)/16
        else: heuristic_red += (1-closest_food/76)/8 - (1-closest_enemy/76)/16
    else:
        if data.player <= 1: heuristic_blue += (1-closest_enemy/76)/8 + (1-closest_capsule/76)/16
        else: heuristic_blue += (1-closest_food/76)/8
    return heuristic_red, heuristic_blue


def removeStop(list:list) -> None:
    try: list.remove('Stop')
    except ValueError: pass
    
    
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
