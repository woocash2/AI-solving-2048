from src.controller import Controller
import src.utils.movesprep as movesprep
from src.utils.movesprep import data_path
from src.board import Board
from threading import Thread
from mcts.mcts import NetworkMCTS
import pickle
import src.utils.representations as repr
import os
import numpy as np


class NetController(Controller):
    def __init__(self):
        super().__init__()

    def make_moves(self):
        lm, rm = movesprep.load_moves()
        with open(os.path.join(data_path(), 'mcts-net.pickle1'), 'rb') as file:
            net = pickle.load(file)
        nmcts = NetworkMCTS(lm, rm, 0., net, 0.997)

        self.board.spawn_random_tile()
        i = 1
        while self.board.move_exists():
            print('move', i)
            i += 1
            self.board.brd = nmcts.best_move_net(self.board.brd, net)
            self.update_view()

        print('finished after', i, 'moves')

    def play_game(self, board):
        self.board = board
        self.create_gui()
        th = Thread(target=self.make_moves)
        th.setDaemon(True)

        th.start()
        self.win.mainloop()


def main(e):
    left_moves, right_moves = movesprep.load_moves()
    board = Board(left_moves, right_moves)
    game = NetController()
    game.play_game(board)


if __name__ == '__main__':
    main(None)
