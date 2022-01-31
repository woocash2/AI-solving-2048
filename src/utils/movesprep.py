import pickle
from pathlib import Path
import os

POSSIBLE_ROWS = 16 * 16 * 16 * 16


"""
if numbers stored in row are 2^a, 2^b, 2^c, 2^d then the bijective hash of this row is:
[a - 1] + [16 * (b - 1)] + [16^2 * (c - 1)] + [16^3 * (d - 1)]
it is a bijection since a, b, c, d vary from 1 to 16 inclusive
"""


def data_path():
    return os.path.join(Path(__file__).parent.parent.parent, 'data')


def row_to_hash(row):
    hsh = 0
    mult = 1

    for val in row:
        hsh += mult * val
        mult *= 16

    return hsh


def hash_to_row(hsh):
    row = []

    while hsh > 0:
        row.append(hsh % 16)
        hsh = int(hsh / 16)
    while len(row) < 4:
        row.append(0)

    return row


def flush_left(row):
    for i in range(len(row)):
        for j in range(len(row)):
            if j > 0 and row[j] > 0 and row[j - 1] == 0:
                row[j - 1] = row[j]
                row[j] = 0


def merge_blocks(row):
    for i in range(len(row) - 1):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i + 1] = 0
            row[i] += 1


def get_move_left(row):
    new_row = row.copy()
    flush_left(new_row)
    merge_blocks(new_row)
    flush_left(new_row)

    return new_row


def precompute_moves():
    left_moves = {}
    right_moves = {}

    for hsh in range(POSSIBLE_ROWS):
        row = hash_to_row(hsh)
        left_row = get_move_left(row)
        left_moves[row_to_hash(row)] = row_to_hash(left_row)

        row.reverse()
        right_row = get_move_left(row)
        row.reverse()
        right_row.reverse()
        right_moves[row_to_hash(row)] = row_to_hash(right_row)

    with open(os.path.join(data_path(), 'left_moves.pickle'), 'wb') as file:
        pickle.dump(left_moves, file)
    with open(os.path.join(data_path(), 'right_moves.pickle'), 'wb') as file:
        pickle.dump(right_moves, file)


def load_moves():
    with open(os.path.join(data_path(), 'left_moves.pickle'), 'rb') as file:
        left_moves = pickle.load(file)
    with open(os.path.join(data_path(), 'right_moves.pickle'), 'rb') as file:
        right_moves = pickle.load(file)
    return left_moves, right_moves


if __name__ == '__main__':
    precompute_moves()
