from funcs import select_partition, normalize, dc_alg, get_partition, segment, \
    auto_args, spread, dc_alg, weighted_dc_alg, print_progress_bar, easy_midi_generator
from funcs import normal_distribution_maker as ndm
import itertools
import numpy as np
from matplotlib import pyplot as plt
import random


class Instrument:
    @auto_args
    def __init__(self, max_td):
        # vars assigned outside of class: num
        self.notes = []

class Player:
    @auto_args
    # advanced players can play faster / cover more instruments. 4?
    # beginning players should have only 2 instruments (or maybe even only 1?)

    def __init__(self, noi, player_num):
        # will prob need to be refined; currently max speed is 7, min is 1
        self.max_td = 2 * (2 ** (self.player_num / 3.5))
        self.insts = [Instrument(self.max_td/self.noi) for i in range(self.noi)]

class Atom:
    @auto_args

    def __init__(self, insts, atom_nCVI, atom_dur):
        #what is the max td? min td? need to get an nCVI and a td -> number of notes per thing and per inst.
        # min number of notes would be "1", max would be decided by max td?
        # each instrument has a max td, but what does that mean for the group as a whole?
        # does each instrument need to have at least 1 item? What about concurrency?
        # Should notes just be dropped randomly?
        # Maybe each section has a different nCVI?

        # assign chord size probs based on number of instruments. Pretty random;
        # but should it lean toward smaller groups? should it be graded somehow?
        if len(self.insts) == 1:
            cs_probs = [1.0]
        else:
            cs_probs = np.random.normal(0.5, 0.25, size = len(self.insts))
            cs_probs = np.clip(cs_probs, 0, 1)
            cs_probs *= (1 / (np.arange(len(self.insts)) + 1)) ** 0.5
            if sum(cs_probs) == 0:
                print('cs_probs is where the zero problem is')
                print(cs_probs)
            cs_probs = normalize(cs_probs)

        nons = []
        # for each instrument, for each atom, you need a td.
        for inst in self.insts:
            # print(inst)
            max_td = inst.max_td
            min_td = 1 / self.atom_dur
            td = np.clip(np.random.normal(0.5, 0.25), 0, 1) * np.log2(max_td / min_td)
            td = min_td * (2 ** td)
            #number of notes
            non = td * self.atom_dur
            nons.append(non)

        nons = np.array(nons)
        # print()
        # print(len(self.insts))
        # print(cs_probs)
        if len(self.insts) == 1:
            self.cs_stream = np.repeat(1, np.sum(nons))
        else:
            avg_notes_per_cs_epoch = np.sum((np.arange(len(self.insts)) + 1) * cs_probs)
            estimated_epochs = np.int(np.ceil(np.sum(nons) / avg_notes_per_cs_epoch)) + 2
            self.cs_stream = weighted_dc_alg(np.arange(len(self.insts))+1, estimated_epochs, weights = cs_probs)
        self.note_stream = weighted_dc_alg(np.arange(len(self.insts)), np.int(np.sum(nons)), weights = nons / np.sum(nons))
        if sum(self.cs_stream) < len(self.note_stream): print('we have a problem')
        self.note_matrix = np.zeros((len(self.insts), len(self.cs_stream)), 'int')
        ct = 0
        for i, item in enumerate(self.cs_stream):
            for note in self.note_stream[ct:ct+item]:
                self.note_matrix[note, i] = 1
            ct += item

        nm = self.note_matrix.T
        proper_indices = np.logical_not(np.all(nm == False, axis = 1))
        self.note_matrix = nm[proper_indices].T
        # print(self.note_matrix)
        # print(np.shape(self.note_matrix))
        self.base_rhythm = segment(np.shape(self.note_matrix)[1], self.atom_nCVI)
        dyn_weights = normalize(np.random.uniform(0, 1, 4))
        self.dyns = weighted_dc_alg(np.arange(4), len(self.cs_stream), weights = dyn_weights)

    def juggle_atom(self, spd):
        return normalize(np.array([spread(i, spd) for i in self.base_rhythm])) * self.atom_dur


class Group:
    @auto_args

    def __init__(
        self, insts, section_dur, atomic_min, rest_ratio, atom_nCVI, start_time
        ):
        self.rest_ratio = spread(self.rest_ratio, 1.25)
        self.make_frame()
        self.atom = Atom(self.insts, self.atom_nCVI, self.atom_dur)
        self.fill_frame()

    def make_frame(self):
        # make atomic size
        log_spread =  np.log2(self.section_dur/self.atomic_min) - 3
        self.atom_dur = self.atomic_min * (2**np.random.uniform(log_spread))
        self.reps = np.int(np.floor((self.section_dur * (1 - self.rest_ratio)) / self.atom_dur))
        self.rep_partition = select_partition(self.reps, type='rounded', one_allowed='no')
        np.random.shuffle(self.rep_partition)
        #number of rests
        self.nor = len(self.rep_partition) - 1
        self.rests = segment(self.nor, 10) * (self.section_dur - (self.reps * self.atom_dur))

    def fill_frame(self):
        st = self.start_time
        self.note_locs = []
        for i, rp in enumerate(self.rep_partition):
            for j in range(rp):
                atom_start = st
                st += self.atom_dur
                atom = self.atom.juggle_atom(1.25)
                for k in range(len(atom)):
                    self.note_locs.append(atom_start + np.sum(atom[:k]))
            if i != len(self.rep_partition)-1:
                st += self.rests[i]
        # print('')
        # print(self.atom.note_matrix)
        # print(self.reps)
        self.group_note_matrix = np.tile(self.atom.note_matrix, self.reps)
        # print(self.group_note_matrix)
        # print('')

        self.group_dyns = np.tile(self.atom.dyns, self.reps)






class Section:
    @auto_args

    def __init__(
        self, section_num, section_dur, atomic_min, rest_ratio, nos, start_time
        ):
        # vars assigned outside of class: noi, inst_stream, insts
        self.atom_nCVI = np.random.uniform(20)

    def __continue__(self):
        self.partition = select_partition(len(self.insts))
        self.make_grouping()
        # print(self.section_num)
        print_progress_bar(self.section_num+1, self.nos, prefix='Progress:', suffix='Complete', length=50)

    def make_grouping(self):
        random.shuffle(self.inst_stream)
        self.groups = []
        ct = 0
        sd = self.section_dur
        am = self.atomic_min
        rr = self.rest_ratio
        an = self.atom_nCVI
        st = self.start_time
        for i, item in enumerate(self.partition):
            # print(item)
            gi = self.insts[ct:ct+item]
            if len(gi) == 0:
                print('whoops')
                print(self.partition)
            # print()
            # print(gi)
            # print(len(self.insts))
            # print()
            # gi = [self.insts[j] for j in gi]
            ct += item
            self.groups.append(Group(gi, sd, am, rr, an, st))





class Piece:
    @auto_args

    def __init__(self, players, nos, dur_tot, atomic_min, avg_rr):
        self.vels = [20, 50, 80, 110]
        self.noi = sum([player.noi for player in players])
        self.section_durs = segment(self.nos, 10) * self.dur_tot
        self.section_start_times = [sum(self.section_durs[:i]) for i in range(self.nos)]
        self.rrs = segment(self.nos, 10) * self.avg_rr * self.nos
        self.assign_inst_nums()
        print_progress_bar(0, self.nos, prefix='Progress:', suffix='Complete', length=50)
        self.instantiate_sections()
        self.make_section_instrumentation()
        # self.view_instrumentation_grid()
        for section in self.sections: section.__continue__()
        self.notes_to_insts()
        self.print_midi()

    def notes_to_insts(self):
        for section in self.sections:
            for group in section.groups:
                for i, inst in enumerate(group.insts):
                    event_indices = np.nonzero(group.group_note_matrix[i])
                    # print()
                    # # print(group.reps)
                    # print(group.group_note_matrix)
                    # print(group.group_note_matrix[:, i])
                    # print(event_indices)
                    # print()
                    group.note_locs = np.array(group.note_locs)
                    note_locs = group.note_locs[event_indices]
                    dyns = group.group_dyns[event_indices]
                    for j, note_loc in enumerate(note_locs):
                        inst.notes.append([inst.num + 44, note_loc, 0.1,self.vels[dyns[j]]])

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
        nos = self.nos
        for i in range(self.nos):
            sd = self.section_durs[i]
            rr = self.rrs[i]
            st = self.section_start_times[i]
            self.sections.append(Section(i, sd, am, rr, nos, st))



# gotta happen at piece level, so that we can dc_alg around, so that there are
# different insts for each section
    def make_section_instrumentation(self):
        # first, how many insts per section? min = 1, max = max; log scale? me thinks yes
        # should try to get even spread? normal curve?
        dist = ndm(self.noi) ** 0.1
        if sum(dist) == 0: print('dist is where the problem is')
        dist = normalize(dist)
        # dist = normalize(ndm(self.noi) ** 0.1)
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

    def print_midi(self):
        path = 'saves/midi/'
        notes = [inst.notes for inst in self.insts]
        notes = [i for i in itertools.chain.from_iterable(notes)]
        # for inst in self.insts:
        #     notes.append(inst.notes)
        #     name = str(inst.num) + '.mid'
        easy_midi_generator(notes, path + 'all_together.mid', 'Trumpet')
