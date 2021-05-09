import numpy as np
import pickle

import moves

LEFT = -1
RIGHT = 1
UP = -2
DOWN = 2


class Board:
    def __init__(self, left_row_moves, right_row_moves):
        self.lrm = left_row_moves
        self.rrm = right_row_moves
        self.tiles = np.zeros(shape=(4, 4), dtype=int)

    def to_string(self):
        s = ''
        for i in range(4):
            for j in range(4):
                s += ' ' + str(self.tiles[i][j])
            s += '\n'
        return s

    def swipe(self, direction):
        before_move = self.tiles.copy()
        if direction == LEFT or direction == RIGHT:
            for i in range(len(self.tiles)):
                hsh = moves.row_to_hash(self.tiles[i])
                new_hsh = self.lrm[hsh] if direction == LEFT else self.rrm[hsh]
                self.tiles[i] = np.array(moves.hash_to_row(new_hsh))
        if direction == UP or direction == DOWN:
            for i in range(len(self.tiles)):
                hsh = moves.row_to_hash(self.tiles[:, i])
                new_hsh = self.lrm[hsh] if direction == UP else self.rrm[hsh]
                self.tiles[:, i] = np.array(moves.hash_to_row(new_hsh))
        comparision = before_move == self.tiles
        return not comparision.all()

    def spawn_random_tile(self):
        empty_tiles = []
        for i in range(4):
            for j in range(4):
                if self.tiles[i][j] == 0:
                    empty_tiles.append((i, j))
        spawn_pos = empty_tiles[np.random.randint(0, len(empty_tiles))]
        tile_distrib = np.random.uniform(0.0, 1.0)
        self.tiles[spawn_pos[0]][spawn_pos[1]] = 1 if tile_distrib < 0.9 else 2

    def move_exists(self):
        for i in range(4):
            hsh_row = moves.row_to_hash(self.tiles[i])
            hsh_col = moves.row_to_hash(self.tiles[:, i])
            if hsh_row != self.lrm[hsh_row] or hsh_row != self.rrm[hsh_row]:
                return True
            if hsh_col != self.lrm[hsh_col] or hsh_col != self.rrm[hsh_col]:
                return True
        return False


if __name__ == '__main__':
    left_moves, right_moves = moves.load_moves()
    board = Board(left_moves, right_moves)
    dirs = {'w': UP, 'a': LEFT, 's': DOWN, 'd': RIGHT}

    while True:
        board.spawn_random_tile()
        print(board.to_string())
        moved = False
        while not moved:
            choice = ''
            while choice not in ['w', 'a', 's', 'd']:
                choice = input()
            moved = board.swipe(dirs[choice])
        print(board.to_string())
