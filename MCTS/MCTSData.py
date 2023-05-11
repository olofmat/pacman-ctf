import numpy as np

from capture import GameState
from MCTS.Node import Node

class MCTSData:    
    def __init__(self, state:GameState, player:int, UCB1:float, sim_time:float = np.inf, sim_number:int = 1_000_000) -> None:
        self.state = state
        self.player = player
        self.UCB1 = UCB1
        self.sim_time = sim_time
        self.sim_number = sim_number
        self.distances = np.zeros(0)
        
        