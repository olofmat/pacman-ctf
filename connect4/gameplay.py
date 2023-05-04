import numpy as np
import random


def availableMoves(currentState: np.ndarray) -> list:
    returnMoves = []
    availableMoves_ = np.arange(7)

    for col in availableMoves_:
        if currentState[0][col] == 0:
            returnMoves.append(int(col))
    return returnMoves


def makeMove(currentState: np.ndarray, player: int, col: int) -> int:
    if type(col) != int:
        return

    for row in range(5, -1, -1):
        if currentState[row][col] == 0:
            currentState[row][col] = player
            return row


def randomMove(moves: list) -> int:
    return random.choice(moves)


def gameEnd(board: np.ndarray, row: int, col: int) -> np.ndarray:
    if type(col) != int or type(row) != int:
        return np.array([0, 0])

    startR = max(row-3, 0)
    endR = min(row+1, 3)
    startC = max(col-3, 0)
    endC = min(col+1, 4)

    # Check horizontal locations for win
    for c in range(startC, endC):
        firstSpot = board[row][c]
        if firstSpot == 0:
            continue
        if firstSpot == board[row][c+1] == board[row][c+2] == board[row][c+3]:
            return winner(firstSpot)

    # Check vertical locations for win
    for r in range(startR, endR):
        firstSpot = board[r][col]
        if firstSpot == 0:
            continue
        if firstSpot == board[r+1][col] == board[r+2][col] == board[r+3][col]:
            return winner(firstSpot)

    # Check negatively sloped diagonals
    for c in range(startC, endC):
        for r in range(startR, endR):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return winner(firstSpot)

    # Check positively sloped diagonals
    for c in range(startC, endC):
        for r in range(startR+3, endR+3):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                return winner(firstSpot)

    return np.array([0, 0])


def winner(player: int) -> np.ndarray:
    if player == 1:
        return np.array([1, -1])
    elif player == -1:
        return np.array([-1, 1])


def nextPlayer(player: int) -> int:
    return -1*player
