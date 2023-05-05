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
from game import Directions
import game
from MCTS.AI import MCTSfindMove


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
    self.startingNumberOfFood = len(self.getFood(gameState).asList())
    
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState:GameState):

    # print("")
    # print(gameState.getScore())

    time.sleep(0.2)

    print("myTeam/choseAction")
    print(self.heuristicFunction(gameState))
    self.heuristicFunction(gameState)
    # if()    
    # print(gameState.getRedFood)
    # print([gameState.getAgentPosition(0),gameState.getAgentPosition(1),gameState.getAgentPosition(2),gameState.getAgentPosition(3)])
    sum = 0
    # print(gameState.getLegalActions((self.index+2)%4))
    # for i in [0,1,2,3]:
    #   if(gameState.getAgentPosition(i) != None):
    #     sum += 1

    # print(sum)
    actions = gameState.getLegalActions(self.index)
    
    # actions.remove("Stop")
    # print(actions)
    # print(actions[0])


    return random.choice(actions)
  
  
  def heuristicFunction(self, gameState:GameState) -> float:

    foodCapturedByYou =  self.startingNumberOfFood-len(self.getFood(gameState).asList())
    foodCapturedByOpponent = self.startingNumberOfFood - len(self.getFoodYouAreDefending(gameState).asList())    
    score = self.getScore(gameState)

    ### RESONABLE HEURISTIC. Maximize your score. Maximize how much you're carrying but less so than how much you deposited.
    ### Minimize how much food your opponent has captured but it's harder so dont spend to much time on it.
    heuristic = score+1/4*foodCapturedByYou - 1/4*foodCapturedByOpponent

    return heuristic



