import numpy as np
import src.utils.movesprep as movesprep


def contains(state, target):
    tiles = tiles_2d(state)
    for x in tiles:
        for y in x:
            if y == target:
                return True
    return False


def transpose(board):
    """
    :param board: list of 4 integers, each representing a single row
    :return: column representation of board (list of 4 integers, each representing a single column)
    """
    tr = [0, 0, 0, 0]
    for j in range(4):
        for i in range(4):
            code = int(board[i] / (16 ** j)) % 16
            tr[j] += code * (16 ** i)
    return tr


def tiles_2d(board):
    """
    :param board: list of 4 integers, each representing a single row
    :return: 2d array of tile values
    """
    tiles = []
    for i in range(4):
        tiles.append(movesprep.hash_to_row(board[i]))
    return tiles


def chanelled_tiles(tiles):
    """
    :param tiles: 2d list of tiles
    :return: numpy array of shape (4, 4, 16) each channel corresponding to a single value of tile
    """
    chanelled = np.zeros((4, 4, 16))
    for i in range(4):
        for j in range(4):
            chanelled[i][j][tiles[i][j]] = 1.
    return chanelled


def chanelled_flat(tiles):
    chan = np.zeros((4, 4, 1))
    for i in range(4):
        for j in range(4):
            chan[i][j][0] = tiles[i][j]
    return chan


if __name__ == '__main__':
    lm, rm = movesprep.load_moves()
    x = [1,1,1,1]
    t = transpose(x)
    print(t)
    yt = []
    y = []
    for i in range(4):
        yt.append(rm[t[i]])
    for i in range(4):
        y.append(lm[x[i]])
    print(y)
    print(yt)

    print('chanel test')
    tiles = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [2, 3, 0, 0],
        [0, 0, 5, 1],
    ]

    h = chanelled_tiles(tiles)
    for c in range(16):
        print('chanel', c)
        for i in range(4):
            print(h[i, :, c])

    from src.mllib.models import Sequential
    from src.mllib.layers import Convolutional
    from src.mllib.layers import Flatten
    from src.mllib.layers import Dense


    model = Sequential()
    model.add_layer(Convolutional(4, 4, 16, 8, 2, 'tanh'))
    model.add_layer(Convolutional(3, 3, 8, 8, 2, 'tanh'))
    model.add_layer(Flatten(2, 2, 8))
    model.add_layer(Dense(32, 16, 'tanh'))
    model.add_layer(Dense(16, 1))
    print(model.predict(np.array([h])))
