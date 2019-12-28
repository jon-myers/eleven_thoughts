from compose import Piece, Player
import numpy as np
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

piece = Piece(players, 10)
print(piece.noi)
