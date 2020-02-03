from compose import Piece, Player
import numpy as np
import pickle
#players:
# advanced: Lucas, Zach, Jon, Henry
# potentials: Ed, Sam
# beginners: Joanne, Jake, nick, nathan?, ku? Assaf?

# start by just assuming everyone, max capacity

lucas = Player(4, 9)
jon = Player(4, 8)
zach = Player(4, 7)
henry = Player(4, 6)
sam = Player(4, 5)
jake = Player(3, 4)
nick = Player(3, 3)
joanne = Player(3, 2)
nathan = Player(3, 1)
assaf = Player(3, 0)
players = [lucas, jon, zach, henry, sam, jake, nick, joanne, nathan, assaf]
num_of_sections = 17
dur_tot = 60 * 9.5
atomic_min = 1.1
avg_rr = 0.2
piece = Piece(players, num_of_sections, dur_tot, atomic_min, avg_rr)

pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
