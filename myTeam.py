# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
from capture import GameState
import random, time, util
from game import Directions, Actions
import game
from MCTS.AI import MCTSfindMove
from MCTS.MCTSData import MCTSData
import heapq
from typing import Tuple
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState:GameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.WALLS = gameState.data.layout.walls
    self.DIR_STR2VEC = {'North':(0,1), 'South':(0,-1), 'East':(1,0), 'West':(-1,0)}
    self.DIR_VEC2STR = {(0,1):'North', (0,-1):'South', (1,0):'East', (-1,0):'West'}
    
    CaptureAgent.registerInitialState(self, gameState)
    self.data = MCTSData(gameState, self.index, UCB1=0.4, sim_time=0.9)
    self.data.distances = self.calculate_distances()
    
    middle = (15, 7)
    self.moves = self.movesToPoint(self.data.state.getAgentPosition(self.index), middle)


  def chooseAction(self, gameState:GameState) -> str:
    self.data.state = gameState
    if self.moves:
        return self.moves.pop(0)
    return MCTSfindMove(self.data)


  def movesToPoint(self, init_pos:Tuple[int], point:Tuple[int]) -> list[str]:
    checked = set()
    queue = []
    
    heapq.heappush(queue, (0, [init_pos]))
    checked.add(init_pos)

    while queue:
        current_cost, current_path = heapq.heappop(queue)
        current_pos = current_path[-1]
        
        if current_pos == point:
            path = np.array(current_path)
            (path[1:] - path[:-1]).tolist()
            
            return [self.DIR_VEC2STR[(current_path[i+1][0] - current_path[i][0], current_path[i+1][1] - current_path[i][1])] for i in range(len(current_path)-1)]

        for dir in self.getPossibleDirections(current_pos):
            dx, dy = self.DIR_STR2VEC[dir]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if new_pos not in checked:
                new_path = current_path.copy()
                new_path.append(new_pos)
                heapq.heappush(queue, (current_cost+1, new_path))
                checked.add(new_pos)
        
        
  def getPossibleDirections(self, pos:Tuple[int]) -> list[str]:
    possible = []
    x, y = pos

    for dir, vec in self.DIR_STR2VEC.items():
        if dir == 'Stop': continue
        dx, dy = vec
        next_y = y + dy
        next_x = x + dx
        if not self.WALLS[next_x][next_y]: possible.append(dir)

    return possible


  def calculate_distances(self) -> np.ndarray:
    distance_matrix = np.zeros((self.WALLS.width, self.WALLS.height, self.WALLS.width, self.WALLS.height))

    for y_start in range(self.WALLS.height):
        for x_start in range(self.WALLS.width):
            if self.WALLS[x_start][y_start]: continue
            for y in range(y_start, self.WALLS.height):
                for x in range(x_start, self.WALLS.width):
                    if (x_start, y_start) == (x, y): continue
                    if self.WALLS[x][y]: continue

                    path_length = len(self.movesToPoint((x_start, y_start), (x, y)))

                    distance_matrix[x_start][y_start][x][y] = path_length
                    distance_matrix[x][y][x_start][y_start] = path_length
    return distance_matrix