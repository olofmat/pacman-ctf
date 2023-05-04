import pygame
import sys
import numpy as np
from MCTS.gameplay import availableMoves

WHITE = (230, 230, 230)
GREY = (180, 180, 180)
PURPLE = (138, 43, 226)
RED = (220, 20, 60)


def initializeGame(WIDTH: int, HEIGHT: int):
    gameState = np.zeros((6, 7))

    pygame.init()
    screen = pygame.display.set_mode((7 * WIDTH, 7 * HEIGHT))
    frame = pygame.Surface((7 * WIDTH, 6 * HEIGHT))
    frame.fill(GREY)
    for i in range(7):
        for j in range(6):
            pygame.draw.circle(frame, WHITE,
                               int2coord(i, j, WIDTH, HEIGHT), WIDTH//3)
    frame.set_colorkey(WHITE)

    return gameState, screen, frame


def draw(screen: pygame.Surface, frame: pygame.Surface, board: np.ndarray, WIDTH: int, HEIGHT: int, move: int = None, player: int = 0) -> None:
    if type(move) == int:
        animatePiece(screen, frame, board, move, player, WIDTH, HEIGHT)

    screen.fill(WHITE)
    drawPieces(screen, board, player, WIDTH, HEIGHT)
    screen.blit(frame, (0, HEIGHT))

    pygame.display.flip()


def drawPieces(screen: pygame.Surface, board: np.ndarray, player, WIDTH: int, HEIGHT: int, lastPlaced: tuple = (None, None)) -> None:
    # placed pieces
    for i, row in enumerate(board):
        for j, spot in enumerate(row):
            if (i, j) == lastPlaced:
                continue
            elif spot == 1:
                pygame.draw.circle(screen, PURPLE,
                                   int2coord(j, i+1, WIDTH, HEIGHT), WIDTH // 3)
            elif spot == -1:
                pygame.draw.circle(screen, RED,
                                   int2coord(j, i+1, WIDTH, HEIGHT), WIDTH // 3)
    color = PURPLE if player == 1 else RED
    # moving piece
    mPos = mousePos(WIDTH)
    mPos = mPos if mPos != None else 3
    pygame.draw.circle(screen, color,
                       int2coord(mPos, 0, WIDTH, HEIGHT), WIDTH // 3)


def animatePiece(screen: pygame.Surface, frame: pygame.Surface, board: np.ndarray, col: int, player: int, w: int, h: int):
    for row in range(6):
        if board[row][col] != 0:
            break

    color = RED if player == 1 else PURPLE
    y = 0
    while y < (row+1)*h:
        screen.fill(WHITE)
        drawPieces(screen, board, player, w, h, (row, col))
        pygame.draw.circle(screen, color,
                           (w*col + w/2, y+h/2), w // 3)
        screen.blit(frame, (0, h))
        y += h*0.012
        pygame.display.flip()


def resolveEvent(board: np.ndarray, player: int, WIDTH: int):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and player:
            return placePiece(board, WIDTH)
    return False


def placePiece(board: np.ndarray, WIDTH: int):
    moves = availableMoves(board)
    mPos = mousePos(WIDTH)

    if mPos in moves:
        return mPos
    return False


def gameOver(screen: pygame.Surface, result: np.ndarray, WIDTH: int) -> bool:
    color = PURPLE if result[0] == 1 else RED
    font = pygame.font.Font(None, 128)
    while True:
        win = font.render("Winner", True, color)
        screen.blit(win, (3.5 * WIDTH - font.size('Winner')[0]/2, WIDTH*0.1))
        pygame.display.flip()
        for event_ in pygame.event.get():
            if event_.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event_.type == pygame.KEYDOWN and event_.key == pygame.K_SPACE:
                return False


def int2coord(i: int, j: int, w: int, h: int) -> np.ndarray:
    return np.array([w*i + w/2, h*j + h/2])


def mousePos(WIDTH: int):
    mPos = pygame.mouse.get_pos()[0]
    for i in range(7):
        if WIDTH * i < mPos <= WIDTH * (i+1):
            return i
    return None


def chooseConfig(SIMULATIONS: int) -> int:
    if len(sys.argv) == 1:
        return SIMULATIONS

    if len(sys.argv) == 2:
        try:
            sims = int(sys.argv[1])
        except ValueError:
            print(
                '\n Usage: \n No arguments; 1000 simulations \n One argument; {Number of simulations (int)}')
            sys.exit()
        return sims

    print(
        '\n Usage: \n No arguments; 1000 simulations \n One argument; {Number of simulations (int)}')
    sys.exit()
