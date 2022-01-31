import tkinter as tk
from controller import INFO
import playercontroller
import stdsearchcontroller
import mctscontroller
import netcontroller
from playercontroller import PlayerController
from stdsearchcontroller import StdSearchController
from mctscontroller import MCTSController


class MainMenu:
    def __init__(self):
        self.win = tk.Tk()
        self.grid = tk.Frame(bg=INFO[-1]['color'])
        self.ctrl = None
        pass

    def set_player_controller(self, event):
        self.win.destroy()
        playercontroller.main(None)

    def set_stdsearch_controller(self, event):
        self.win.destroy()
        stdsearchcontroller.main(None)

    def set_mcts_controller(self, event):
        self.win.destroy()
        mctscontroller.main(None)

    def set_net_controller(self, event):
        self.win.destroy()
        netcontroller.main(None)

    def show_modes(self):
        player_label = tk.Label(
            master=self.grid,
            bg=INFO[0]['color'],
            font=('Helvetica', 36),
            text='Play',
            width=10,
            height=2
        )
        player_label.grid(row=0, column=0, padx=10, pady=10)
        player_label.bind("<Button-1>", self.set_player_controller)

        heur_label = tk.Label(
            master=self.grid,
            bg=INFO[0]['color'],
            font=('Helvetica', 36),
            text='std search',
            width=10,
            height=2
        )
        heur_label.grid(row=1, column=0, padx=10, pady=10)
        heur_label.bind("<Button-1>", self.set_stdsearch_controller)

        mcts_label = tk.Label(
            master=self.grid,
            bg=INFO[0]['color'],
            font=('Helvetica', 36),
            text='MCTS',
            width=10,
            height=2
        )
        mcts_label.grid(row=2, column=0, padx=10, pady=10)
        mcts_label.bind("<Button-1>", self.set_mcts_controller)

        net_label = tk.Label(
            master=self.grid,
            bg=INFO[0]['color'],
            font=('Helvetica', 36),
            text='Network',
            width=10,
            height=2
        )
        net_label.grid(row=3, column=0, padx=10, pady=10)
        net_label.bind("<Button-1>", self.set_net_controller)

        self.grid.pack()
        self.win.mainloop()


if __name__ == '__main__':
    main_menu = MainMenu()
    main_menu.show_modes()
