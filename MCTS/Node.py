import random
import numpy as np
from capture import GameState


class Node:
    def __init__(self, player: int, move:str=None, parent=None) -> None:
        self.value = np.array(0, np.float32)
        self.visits = np.array(0, np.int32)
        self.parent = parent
        self.children = []
        self.move = move
        self.player = player                    # self.player makes self.move


    def makeChildren(self, player: int, moves: list) -> None:
        """Makes a child node for every possible move"""
        for move in moves:
            child = Node(player, move, parent=self)
            self.children.append(child)

        random.shuffle(self.children)


    def selectChild(self, C: float) -> object:
        """Uses UCB1 to pick child node"""
        # if node doesn't have children, return self
        if len(self.children) == 0:
            return self

        UCB1values = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            # returns child if it hasn't been visited before
            if child.visits == 0:
                return child

            # calculates UCB1
            v = child.value
            mi = child.visits
            mp = child.parent.visits
            UCB1values[i] = v/mi + C * np.sqrt(np.log(mp)/mi)

        # return child that maximizes UCB1
        maxIndex = np.argmax(UCB1values)
        return self.children[maxIndex]


    def backpropagate(self, gameState:GameState, result: float) -> None:
        """Updates value and visits according to result"""
        instance = self
        while instance != None:
            instance.visits += 1
            instance.value += result if gameState.isOnRedTeam(instance.player) else -result
            instance = instance.parent


    def chooseMove(self) -> int:
        """Chooses most promising move from the list of children"""
        # if node doesn't have children, make no move
        if len(self.children) == 0:
            return self.move

        # finds child with most visits and returns it
        visits = [child.visits for child in self.children]
        maxVisits = max(visits)
        maxIndex = visits.index(maxVisits)

        chosenChild = self.children[maxIndex]
        return chosenChild.move


    def nextPlayer(self, players:np.ndarray) -> int:
        player_index = np.where(players == self.player)        
        return (player_index[0][0]+1) % players.size
    