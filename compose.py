from funcs import select_partition, normalize, dc_alg, get_partition, segment, \
    auto_args, spread
from funcs import normal_distribution_maker as ndm

import numpy as np
from matplotlib import pyplot as plt
import random


class Instrument:
    @auto_args
    def __init__(self, max_td):
        # vars assigned outside of class: num
        self.place_holder = 'nothing'

class Player:
    @auto_args
    # advanced players can play faster / cover more instruments. 4?
    # beginning players should have only 2 instruments (or maybe even only 1?)

    def __init__(self, noi, player_num):
        # will prob need to be refined; currently max speed is 7, min is 1
        self.max_td = 2 ** (self.player_num / 3.5)
        self.insts = [Instrument(self.max_td/self.noi) for i in range(self.noi)]

# class Phrase:
#     @auto_args
#
#     def __init__(self):


class Group:
    @auto_args

    def __init__(self, insts, section_dur, atomic_min, rest_ratio):
        self.rest_ratio = spread(self.rest_ratio, 1.25)
        self.make_frame()

    def make_frame(self):
        # make atomic size
        log_spread =  np.log2(self.section_dur/self.atomic_min) - 3
        self.atomic_size = self.atomic_min * (2**np.random.uniform(log_spread))
        self.reps = np.int(np.floor((self.section_dur * (1 - self.rest_ratio)) / self.atomic_size))
        self.rep_partition = select_partition(self.reps, type='rounded', one_allowed='no')
        np.random.shuffle(self.rep_partition)
        #number of rests
        self.nor = len(self.rep_partition) - 1
        self.rests = segment(self.nor, 10) * (self.section_dur - (self.reps * self.atomic_size))

        #how many breaks?


class Section:
    @auto_args

    def __init__(self, section_num, section_dur, atomic_min, rest_ratio):
        # vars assigned outside of class: noi, inst_stream, insts
        self.place_holder = 'nothing'

    def __continue__(self):
        self.partition = select_partition(len(self.insts))
        self.make_grouping()

    def make_grouping(self):
        random.shuffle(self.inst_stream)
        self.groups = []
        ct = 0
        # print(self.partition)
        for i, item in enumerate(self.partition):
            # print(item)
            group_insts = self.inst_stream[ct:ct+item]
            ct += item
            self.groups.append(Group(group_insts, self.section_dur, self.atomic_min, self.rest_ratio))



class Piece:
    @auto_args

    def __init__(self, players, nos, dur_tot, atomic_min, avg_rr):
        self.noi = sum([player.noi for player in players])
        self.section_durs = segment(self.nos, 10) * self.dur_tot
        self.rrs = segment(self.nos, 10) * self.avg_rr * self.nos
        self.assign_inst_nums()
        self.instantiate_sections()
        self.make_section_instrumentation()
        # self.view_instrumentation_grid()
        for section in self.sections: section.__continue__()

    def assign_inst_nums(self):
        ct = 0
        self.insts=[]
        for player in self.players:
            for inst in player.insts:
                inst.num = ct
                ct += 1
                self.insts.append(inst)

    def instantiate_sections(self):
        self.sections = []
        am = self.atomic_min
        for i in range(self.nos):
            sd = self.section_durs[i]
            rr = self.rrs[i]
            self.sections.append(Section(i, sd, am, rr))



# gotta happen at piece level, so that we can dc_alg around, so that there are
# different insts for each section
    def make_section_instrumentation(self):
        # first, how many insts per section? min = 1, max = max; log scale? me thinks yes
        # should try to get even spread? normal curve?
        dist = normalize(ndm(self.noi) ** 0.1)
        for section in self.sections:
            section.noi = np.random.choice(np.arange(self.noi), p=dist)
        stream_size = sum([section.noi for section in self.sections])
        inst_stream = dc_alg(np.arange(self.noi), stream_size, alpha=1.0)
        ct = 0
        for section in self.sections:
            section.inst_stream = inst_stream[ct:ct+section.noi]
            ct += section.noi
            section.insts = [self.insts[i] for i in section.inst_stream]

    def view_instrumentation_grid(self):
        grid = np.zeros((self.nos, self.noi), int)
        for section in self.sections:
            print(section.inst_stream)
            grid[section.section_num][section.inst_stream] = 1
        print(grid)
        plt.imshow(grid)
        plt.show()
