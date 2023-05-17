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


import heapq
import random
import sys
import time
from typing import Tuple

import numpy as np

import game
import util
from capture import GameState
from captureAgents import CaptureAgent
from game import Actions, Directions
from MCTS.AI import MCTSfindMove, removeStop
from MCTS.MCTSData import MCTSData
from MCTS.Node import Node

distributions = []

np.set_printoptions(threshold=sys.maxsize)

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




  # ESTIMATING
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
    global distributions
    
    """
    This method handles the initial setup of thegameState.getLegalActions(self.data.player)
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
    self.data = MCTSData(gameState, self.index, UCB1=0.4, sim_time=0.4)
    # self.data.distances = self.calculate_distances()
    # np.save("distances.npy", self.data.distances)
    self.data.distances = np.load("distances.npy")
    
    self.start_pos = gameState.getAgentPosition(self.index)
    self.middle = (10, 7)
    self.moves = self.movesToPoint(self.start_pos, self.middle)


    for index in self.getTeam(gameState):
       if index != self.index: self.friend_index = index

    if self.index in [0,1]: ### ONLY THE FIRST AGENT SHOULD INITIALIZE
      for opponent in self.getOpponents(gameState):
        position = gameState.getInitialAgentPosition(opponent)
        distributions.append(util.Counter())
        distributions[-1][position] = 1
        
    self.enemypositions = [[None, None],[None,None]] # [enemy][0 = current, 1 = past]

    

  def chooseAction(self, gameState:GameState) -> str:
    
    
    my_pos = gameState.getAgentPosition(self.data.player)
    friend_pos = gameState.getAgentPosition(self.friend_index)

    self.update_distributions(gameState,my_pos, friend_pos)
    

    for i_0_1, i in enumerate(self.getOpponents(gameState)):
       self.enemypositions[i_0_1][1] = self.enemypositions[i_0_1][0]
       self.enemypositions[i_0_1][0] = gameState.getAgentPosition(i)
       if self.enemypositions[i_0_1][1] == None:
          continue
       
       if self.enemypositions[i_0_1][0] == None and util.manhattanDistance(self.enemypositions[i_0_1][1],my_pos) <= 2 and my_pos != gameState.getInitialAgentPosition(self.index):
          position = gameState.getInitialAgentPosition(i)
          distributions[i_0_1] =  util.Counter()
          distributions[i_0_1][position] = 1
          self.spread_distribution_for_enemy(gameState,i_0_1,my_pos)
           
    players = []
    
    for i in range(4):
        pos = gameState.getAgentPosition(i)
        if not pos: continue
        threshold = -1 if gameState.isOnRedTeam(self.index) == gameState.isOnRedTeam(i) else 10
        if self.data.player == i or self.data.distances[pos[0]][pos[1]][my_pos[0]][my_pos[1]] <= threshold: players.append(i)
    players = np.array(players)
    self.data.players = players
    
    if self.data.only_me_in_tree and len(players) == 1:
        self.keep_tree()
    else:
        self.discard_tree(gameState)
    self.data.only_me_in_tree = True if len(players) == 1 else False

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


  def keep_tree(self):
    self.data.root = self.data.root.chooseBestChild()
    self.data.root.parent = None
    self.data.state = self.data.state.generateSuccessor(self.data.player, self.data.root.move)

  def discard_tree(self, gameState:GameState):
    self.data.state = gameState
    self.data.root = Node(self.data.player)
    moves = gameState.getLegalActions(self.data.player)
    removeStop(moves)
    self.data.root.makeChildren(self.data.player, moves)

  def calculate_distances(self) -> np.ndarray:
    distance_matrix = np.zeros((self.WALLS.width, self.WALLS.height, self.WALLS.width, self.WALLS.height), np.int32)

    for y_start in range(self.WALLS.height):
        for x_start in range(self.WALLS.width):
            if self.WALLS[x_start][y_start]: continue
            for y in range(self.WALLS.height):
                for x in range(self.WALLS.width):
                    if (x_start, y_start) == (x, y): continue
                    if self.WALLS[x][y]: continue
                    path_length = len(self.movesToPoint((x_start, y_start), (x, y)))
                    distance_matrix[x_start][y_start][x][y] = path_length
                    distance_matrix[x][y][x_start][y_start] = path_length
    
    return distance_matrix
  
  def update_distributions(self,gameState:GameState,my_pos, friend_pos): ### ADD FRIEND
    global distributions
    distances = gameState.getAgentDistances()
    
    
    enemies = self.getOpponents(gameState)

    ### CONVERTS THEIR 0-3 INDEX TO 0-1
    lastenemy = (self.index-1)%4
    if lastenemy <2: enemy_index = 0
    else: enemy_index =1
        

    distribution = distributions[enemy_index]

    if gameState.getAgentPosition(enemies[enemy_index]) != None: ### THEN WE CAN SEE THEM AND KNOW THEIR EXACT POSITION
        distributions[enemy_index] = util.Counter()
        distributions[enemy_index][gameState.getAgentPosition(enemies[enemy_index])] = 1
    else:  
       self.spread_distribution_for_enemy(gameState, enemy_index, my_pos)

    ### NOW GOES THROUGH BOTH ENEMY AGENTS AND PRUNES THEIR POSITIONS BASED ON THE MEASUREMENT
    for i, distribution in enumerate(distributions):
        measured_distance = distances[enemies[i]]
        for position in distribution.keys():
           if distributions[i][position] == 1:
              truedistance = int(util.manhattanDistance(my_pos, position))
              frienddistance = int(util.manhattanDistance(friend_pos, position))
              if(((truedistance <= 5 or frienddistance <= 5) and gameState.getAgentPosition(enemies[i]) != position) or\
                   gameState.getDistanceProb(measured_distance,truedistance) == 0): ### If we can see the square or if the probability of the measurement is zero
                  distributions[i][position] = 0
        
    
    self.displayDistributionsOverPositions(distributions)


  def spread_distribution_for_enemy(self,gameState, enemy_0_1,my_pos):
    global distributions
    distances = gameState.getAgentDistances()
    distribution = distributions[enemy_0_1]
    enemies = self.getOpponents(gameState)
    measured_distance = distances[enemies[enemy_0_1]]
    positions_to_add = []
    for position in distribution.keys():
      if distributions[enemy_0_1][position] == 1: ### IF position IS A POSSIBLE LOCATION FOR THE AGENT
        for direction in self.DIR_VEC2STR:
          new_pos = (position[0]+direction[0],position[1]+direction[1])
          if (new_pos[0]>=0 and new_pos[0] < gameState.data.layout.width) and\
              (new_pos[1]>=0 and new_pos[1] < gameState.data.layout.height) and not gameState.hasWall(new_pos[0],new_pos[1]):  ### NOT OUT OF BOUNDS AND NOT A WALL
            truedistance = int(util.manhattanDistance(my_pos, new_pos))
            if(gameState.getDistanceProb(truedistance,measured_distance) > 0):  ### IF THE MEASUREMENT DID NOT MAKE THAT POSITION IMPOSSIBLE     
              positions_to_add.append(new_pos)
    for newPosition in positions_to_add:
      distributions[enemy_0_1][newPosition] = 1
    self.displayDistributionsOverPositions(distributions)
