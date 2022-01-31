import numpy as np
from src.utils import moves
from src.utils import representations


class Board:
    def __init__(self, left_moves, right_moves):
        """
        :param left_moves: dict mapping row hashes to row hashes after swiping left
        :param right_moves: dict mapping row hashes to row hashes after swiping right
        """
        self.lm = left_moves
        self.rm = right_moves
        self.brd = [0, 0, 0, 0]

    def tiles(self):
        """
        :return: 2d array representation of board
        """
        return representations.tiles_2d(self.brd)

    def to_string(self):
        tiles = self.tiles()
        s = ''
        for i in range(4):
            for j in range(4):
                s += ' ' + str(tiles[i][j])
            s += '\n'
        return s

    def swipe(self, direction):
        """
        :param direction: direction of swipe
        :return: False if move could not happen, True otherwise
        """
        reachable = moves.possible_moves(self.brd, self.lm, self.rm)
        if len(reachable) == 0:
            return False
        if direction not in reachable:
            return False
        self.brd = reachable[direction]
        return True

    def spawn_random_tile(self):
        """
        Chooses an empty tile uniformly at random and then value: 2 with probability 0.9 or 4 with 0.1
        :return: False if could not spawn, True otherwise
        """
        nxt = moves.random_tile(self.brd)
        if not nxt:
            return False
        self.brd = nxt
        return True

    def move_exists(self):
        return len(moves.possible_moves(self.brd, self.lm, self.rm)) > 0


if __name__ == '__main__':
    left_moves, right_moves = moves.load_moves()
    board = Board(left_moves, right_moves)
    dirs = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}

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
