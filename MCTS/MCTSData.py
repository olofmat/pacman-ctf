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
        self.distributions:list[dict]
        self.max_distance:int
        self.defender_threshold:int
        
        self.index_mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        

    def get_food_locations(self) -> None:
        got_food = self.state.getBlueFood() if self.state.isOnRedTeam(self.player) else self.state.getRedFood()
        self.food = []
        for x in range(got_food.width):
            for y in range(got_food.height):
                if got_food[x][y]: self.food.append((x,y))
        
        
    def calculate_threshold(self) -> None:
        if self.state.isOnRedTeam(self.player): score = self.state.getScore()
        else: score = -self.state.getScore()

        self.defender_threshold = 1
        if score > 5:
            self.defender_threshold = 10
