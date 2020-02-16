from compose import Piece, Player
import numpy as np
import pickle

# start by just assuming everyone, max capacity

players = [Player(6, i) for i in range(3)]
num_of_sections = 11
dur_tot = 60 * 9.5
atomic_min = 1.1
avg_rr = 0.2
piece = Piece(players, num_of_sections, dur_tot, atomic_min, avg_rr)

pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
