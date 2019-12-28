from funcs import select_partition


class Player:
    # advanced players can play faster / cover more instruments. 4?
    # beginning players should have only 2 instruments (or maybe even only 1?)

    def __init__(self, num_of_instruments, player_num):
        self.noi = num_of_instruments
        self.player_num = player_num
        # will prob need to be refined; currently max speed is 7, min is 1
        self.max_td = 2 ** (player_num / 3.5)

class Piece:

    def __init__(self, players, num_of_sections):
        self.players = players
        self.nos = num_of_sections
        self.noi = sum([player.noi for player in players])
