import numpy as np
from src.utils.representations import transpose


def possible_moves(board, lm, rm):
    """
    :param board: list of 4 integers, each representing a single row
    :param lm: dict of left moves for rows
    :param rm: dict of right moves for rows
    :return: dict of possible moves labeled by direction
    """
    moves = {}
    nxt = [0, 0, 0, 0]

    for i in range(4):
        nxt[i] = rm[board[i]]
    if nxt != board:
        moves['right'] = nxt.copy()

    for i in range(4):
        nxt[i] = lm[board[i]]
    if nxt != board:
        moves['left'] = nxt.copy()

    transp = transpose(board)
    for i in range(4):
        nxt[i] = rm[transp[i]]
    if nxt != transp:
        moves['down'] = transpose(nxt)

    for i in range(4):
        nxt[i] = lm[transp[i]]
    if nxt != transp:
        moves['up'] = transpose(nxt)

    return moves


def possible_spawns(board):
    """
    :return: returns list of pairs: (board after spawn, probability of this event)
    """
    reachable = []
    num_of_empty = 0

    for i in range(4):
        for j in range(4):
            tile = (int(board[i] / (16 ** j))) % 16
            if tile == 0:
                num_of_empty += 1

    for i in range(4):
        for j in range(4):
            tile = (int(board[i] / (16 ** j))) % 16
            if tile == 0:
                next = board.copy()
                next[i] += 16 ** j
                reachable.append((next.copy(), 9. / (10. * num_of_empty)))
                next[i] += 16 ** j
                reachable.append((next, 1. / (10. * num_of_empty)))

    return reachable


def winning(board, winning_tile=11):
    """
    :param board: list of 4 integers
    :return: True if winning, False otherwise
    """
    for i in range(4):
        for j in range(4):
            code = (board[i] / (16 ** j)) % 16
            if code >= winning_tile:
                return True
    return False


def random_tile(board):
    """
        Chooses an empty tile uniformly at random and then value: 2 with probability 0.9 or 4 with 0.1
        :return: False if could not spawn, Otherwise a board with spawned tile.
        """
    reachable = possible_spawns(board)
    if len(reachable) == 0:
        return False
    nxt = reachable[0][0]
    x = np.random.uniform(0., 1.)
    sum = 0.
    for r in reachable:
        prob = r[1]
        if sum + prob >= x:
            nxt = r[0]
            break
        sum += prob
    return nxt


if __name__ == '__main__':
    x = [0,16**3 * 10 + 16**2 * 5 + 16**1 * 9,0,1]
    print(winning(x, 11))