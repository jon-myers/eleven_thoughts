from compose import Piece, Player
import numpy as np
import pickle
#players:
# advanced: Lucas, Zach, Jon, Henry
# potentials: Ed, Sam
# beginners: Joanne, Jake, nick, nathan?, ku? Assaf?

# start by just assuming everyone, max capacity

lucas = Player(5, 11)
jon = Player(5, 10)
zach = Player(4, 9)
henry = Player(4, 8)
ed = Player(4, 7)
sam = Player(3, 6)
jake = Player(3, 5)
nick = Player(3, 4)
joanne = Player(2, 3)
nathan = Player(2, 2)
ku = Player(2, 1)
assaf = Player(2, 0)
players = [lucas, jon, zach, ed, sam, joanne, jake, nick, nathan, ku, assaf]
num_of_sections = 37
dur_tot = 60 * 17
atomic_min = 0.75
avg_rr = 0.2
piece = Piece(players, num_of_sections, dur_tot, atomic_min, avg_rr)

pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
