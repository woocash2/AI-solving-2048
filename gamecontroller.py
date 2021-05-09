import tkinter as tk
import logic
from logic import LEFT, RIGHT, UP, DOWN
import moves


INFO = {
   -1: {'color': '#646464', 'label': '', 'fontsize': 36},
    0: {'color': '#969696', 'label': '', 'fontsize': 36},
    1: {'color': '#F0F0F0', 'label': '2', 'fontsize': 36},
    2: {'color': '#D2D2D2', 'label': '4', 'fontsize': 36},
    3: {'color': '#FFDC82', 'label': '8', 'fontsize': 36},
    4: {'color': '#FFBE97', 'label': '16', 'fontsize': 36},
    5: {'color': '#FF824B', 'label': '32', 'fontsize': 36},
    6: {'color': '#FF4703', 'label': '64', 'fontsize': 36},
    7: {'color': '#FFFFC6', 'label': '128', 'fontsize': 36},
    8: {'color': '#FFFF8A', 'label': '256', 'fontsize': 36},
    9: {'color': '#FFFF53', 'label': '512', 'fontsize': 36},
    10: {'color': '#FFDD3B', 'label': '1024', 'fontsize': 27},
    11: {'color': '#FFD200', 'label': '2048', 'fontsize': 27},
    12: {'color': '#DEFFFF', 'label': '4096', 'fontsize': 27},
    13: {'color': '#C6FFFF', 'label': '8192', 'fontsize': 27},
    14: {'color': '#86FFFF', 'label': '16384', 'fontsize': 21},
    15: {'color': '#00C7FF', 'label': '32768', 'fontsize': 21}
}


class GameController:
    def __init__(self):
        self.win = tk.Tk()
        self.grid = tk.Frame(bg=INFO[-1]['color'])
        self.frames = []
        self.labels = []
        self.board = None

    def create_gui(self):
        self.frames = []
        self.labels = []

        for i in range(4):
            self.frames.append([])
            self.labels.append([])
            for j in range(4):
                self.frames[i].append(tk.Frame(
                    master=self.grid,
                    bg=INFO[0]['color'],
                    width=100,
                    height=100
                ))
                self.labels[i].append(tk.Label(
                    master=self.grid,
                    bg=INFO[0]['color'],
                    font=('Helvetica', 36),
                ))

        for i in range(4):
            for j in range(4):
                self.frames[i][j].grid(row=i, column=j, padx=10, pady=10)
                self.labels[i][j].grid(row=i, column=j, padx=10, pady=10)

        self.grid.pack()

    def update_view(self):
        self.board.spawn_random_tile()
        for i in range(4):
            for j in range(4):
                value = self.board.tiles[i][j]
                self.frames[i][j]['bg'] = INFO[value]['color']
                self.labels[i][j]['bg'] = INFO[value]['color']
                self.labels[i][j]['text'] = INFO[value]['label']
                self.labels[i][j]['font'] = ('Helvetica', INFO[value]['fontsize'])

    def make_move(self, event):
        moved = False
        if event.keysym == 'Left':
            moved = self.board.swipe(LEFT)
        elif event.keysym == 'Right':
            moved = self.board.swipe(RIGHT)
        elif event.keysym == 'Up':
            moved = self.board.swipe(UP)
        elif event.keysym == 'Down':
            moved = self.board.swipe(DOWN)
        if moved:
            self.update_view()

    def play_game(self, board):
        self.board = board
        self.win.bind('<Left>', self.make_move)
        self.win.bind('<Right>', self.make_move)
        self.win.bind('<Up>', self.make_move)
        self.win.bind('<Down>', self.make_move)
        self.create_gui()
        self.update_view()
        self.win.mainloop()


if __name__ == '__main__':
    left_moves, right_moves = moves.load_moves()
    board = logic.Board(left_moves, right_moves)
    game = GameController()
    game.play_game(board)