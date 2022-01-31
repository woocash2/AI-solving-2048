from src.controller import Controller
import src.utils.movesprep as movesprep
from src.board import Board
from src.mcts.mcts import Search
from threading import Thread
import time


class StdSearchController(Controller):
    def __init__(self):
        super().__init__()

    def make_moves(self):
        lm, rm = movesprep.load_moves()
        std_search = Search(lm, rm)

        self.board.spawn_random_tile()
        i = 1
        while self.board.move_exists():
            print('move', i)
            i += 1
            self.board.brd = std_search.best_move(self.board.brd.copy(), 20)
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
    game = StdSearchController()
    game.play_game(board)


if __name__ == '__main__':
    main(None)
