import numpy as np

from capture import GameState
from MCTS.Node import Node

class MCTSData:    
    def __init__(self, state:GameState, player:int, UCB1:float, sim_time:float = np.inf, sim_number:int = 1_000_000, cutoff:int = 0) -> None:
        self.state = state
        self.player = player
        self.UCB1 = UCB1
        self.sim_time = sim_time
        self.sim_number = sim_number
        self.cutoff = cutoff
        
        self.root:Node
        self.players:np.ndarray
        self.distances: np.ndarray
        self.food:list
        
        self.only_me_in_tree = False


    def get_food_locations(self) -> None:
        got_food = self.state.getBlueFood() if self.state.isOnRedTeam(self.player) else self.state.getRedFood()
        self.food = []
        for x in range(got_food.width):
            for y in range(got_food.height):
                if got_food[x][y]: self.food.append((x,y))
        