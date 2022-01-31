from controller import Controller
import src.utils.movesprep as movesprep
from src.board import Board


class PlayerController(Controller):
    def __init__(self):
        super().__init__()

    def play_game(self, board):
        self.board = board
        self.win.bind('<Left>', self.make_move)
        self.win.bind('<Right>', self.make_move)
        self.win.bind('<Up>', self.make_move)
        self.win.bind('<Down>', self.make_move)
        self.create_gui()
        self.update_view()
        self.win.mainloop()


def main(e):
    left_moves, right_moves = movesprep.load_moves()
    board = Board(left_moves, right_moves)
    game = PlayerController()
    game.play_game(board)


if __name__ == '__main__':
    main(None)
