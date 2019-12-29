from compose import Piece, Player
import numpy as np
import pickle
#players:
# advanced: Lucas, Zach, Jon, Henry
# potentials: Ed, Sam
# beginners: Joanne, Jake, nick, nathan?, ku? Assaf?

# start by just assuming everyone, max capacity

lucas = Player(5, 0)
jon = Player(5, 1)
zach = Player(4, 2)
henry = Player(4, 3)
ed = Player(4, 4)
sam = Player(3, 5)
jake = Player(3, 6)
nick = Player(3, 7)
joanne = Player(2, 8)
nathan = Player(2, 9)
ku = Player(2, 10)
assaf = Player(2, 11)
players = [lucas, jon, zach, ed, sam, joanne, jake, nick, nathan, ku, assaf]
num_of_sections = 17
dur_tot = 60 * 17
atomic_min = 0.75
avg_rr = 0.2
piece = Piece(players, num_of_sections, dur_tot, atomic_min, avg_rr)

pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
