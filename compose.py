from funcs import select_partition, normalize, dc_alg, get_partition, segment, \
    auto_args, spread, dc_alg, weighted_dc_alg, print_progress_bar, \
    easy_midi_generator, bpm_to_pulse_dur, delta_to_pulse_loc, pulses_to_measures, \
    to_time_sig, lp_line_pos
from funcs import normal_distribution_maker as ndm
import itertools
import numpy as np
from matplotlib import pyplot as plt
import random
from quantize import iter_pc, Pulse_Collection



# this is solely for notation. inst_num is relative to the player, not the whole
# shebang.
class Note:

    def __init__(self, lp_note, inst_num):
        self.delta = lp_note[0]
        self.inst_num = inst_num
        self.chord = [inst_num]





# this is solely for notation. inst_num is relative to the player, not the whole
# shebang.
class Measure:
    #pulse_dur is number of pulses in measure
    def __init__(self, pulse_size, pulse_start, noi):
        self.noi = noi
        self.pulse_size = pulse_size
        self.pulse_start = pulse_start
        self.pulse_end = pulse_start + pulse_size
        self.pulses = [1 for i in range(np.int(np.ceil(self.pulse_size)))]
        if self.pulse_size % 1 == 0.5: self.pulses[-1] = 0.5
        self.notes = []

    def remove_duplicates(self):
        for n in range(1, len(self.notes)-1)[::-1]:
            na = self.notes[n]
            nb = self.notes[n-1]
            if na.delta == nb.delta and na.inst_num == nb.inst_num:
                self.notes.pop(n)

    def chordify(self):
        for n in range(1, len(self.notes)-1)[::-1]:
            na = self.notes[n]
            nb = self.notes[n-1]
            if na.delta == nb.delta:
                nb.chord.append(na.inst_num)
                self.notes.pop(n)

    def pulsify(self):
        for note in self.notes:
            note.pulse_delta = note.delta - self.pulse_start


    def get_note_name(self, inst_num):
        if self.noi == 2:
            return ["b", "cis'"][inst_num]
        elif self.noi == 3:
            return ["bes", "c'", "d'"][inst_num]
        elif self.noi == 4:
            return ["a", "b", "cis'", "d'"][inst_num]
        elif self.noi == 5:
            return ["aes", "bes", "c'", "d'", "e'"][inst_num]

    def notate(self):
        out = ''
        for p, pulse in enumerate(self.pulses):
            rel_notes = [n for n in self.notes if n.pulse_delta//1 == p]
            rel_deltas = [np.around(n.pulse_delta%1, decimals=2) for n in rel_notes]
            nn = [self.get_note_name(n.inst_num) for n in rel_notes]
            if pulse == 0.5:
                if len(rel_notes) == 0:
                    out += 'r8 '
                elif len(rel_notes) == 1:
                    if rel_deltas[0] == 0:
                        out += nn[0] + '8 '
                    elif rel_deltas[0] == 0.5:
                        out += 'r16[ ' + nn[0] + '16]'
                elif len(rel_notes) == 2:
                    for i, rd in enumerate(rel_deltas):
                        if i == 0:
                            out += nn[0]+'16[ '
                        if i == 1:
                            out += nn[1] + '16] '
            else:
                if len(rel_notes) == 0:
                    out += 'r4 '
                elif len(rel_notes) == 1:
                    rd = rel_deltas[0]
                    print(rd)
                    if rd == 0:
                        out += nn[0] + '4 '
                    elif rd == 0.25:
                        out += 'r16[ ' + nn[0] + '8.] '
                    elif rd == 0.5:
                        out += 'r8[ ' + nn[0] + '8] '
                    elif rd == 0.75:
                        out += 'r8.[ ' + nn[0] + '16] '
                    else:
                        out += '\\times 2/3 { '
                        if rd == 0.33:
                            out += 'r8 ' + nn[0] + '4 '
                        elif rd == 0.67:
                            out += 'r4 ' + nn[0] + '8 '
                        out += '} '
                elif len(rel_notes) == 2:
                    if rel_deltas[0] == 0:
                        if rel_deltas[1] == 0.25:
                            out += nn[0] + '16[ ' + nn[1] + '8.] '
                        elif rel_deltas[1] == 0.5:
                            out += nn[0] + '8[ ' + nn[1] + '8] '
                        elif rel_deltas[1] == 0.75:
                            out += nn[0] + '8.[ ' + nn[1] + '16] '
                        else:
                            out += '\\times 2/3 { '
                            if rel_deltas[1] == 0.33:
                                out += nn[0] + '8 ' + nn[1] + '4 '
                            elif rel_deltas[1] == 0.67:
                                out += nn[0] + '4 ' + nn[1] + '8 '
                            out += '} '
                    elif rel_deltas[0] == 0.25:
                        if rel_deltas[1] == 0.5:
                            out += 'r16[ ' + nn[0] + '16 ' + nn[1] + '8] '
                        elif rel_deltas[1] == 0.75:
                            out += 'r16[ ' + nn[0] + '8 ' + nn[1] + '16] '
                    elif rel_deltas[0] == 0.5:
                        out += 'r8[ ' + nn[0] + '16 ' + nn[1] + '16] '
                    else:
                        out += '\\times 2/3 { r8[ ' + nn[0] + 'r8 ' + nn[1] + 'r8] } '
                elif len(rel_notes) == 3:
                    if rel_deltas[0] == 0:
                        if rel_deltas[1] == 0.25:
                            if rel_deltas[2] == 0.5:
                                out +=  nn[0] + '16[ ' + nn[1] + '16 ' + nn[2] + '8] '
                            else:
                                out += nn[0] + '16[ ' + nn[1] + '8 ' + nn[2] + '16] '
                        elif rel_deltas[1] == 0.5:
                            out += nn[0] + '8[ ' + nn[1] + '16 ' + nn[2] + '16] '
                        else:
                            out += '\\times 2/3 { ' + nn[0] + '8[ ' + nn[1] + '8 ' + nn[2] + '8] } '
                    elif rel_deltas[0] == 0.25:
                        out += 'r16[ ' + nn[0] + '16 ' + nn[1] + '16 ' + nn[2] + '16] '
                else:
                    out += nn[0] + '16[ ' + nn[1] + '16 ' + nn[2] + '16 ' + nn[3] +'16] '
        out += '|\n'
        return out





















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
        self.max_td = 1 * (2 ** (self.player_num / 3.5))
        self.insts = [Instrument(self.max_td/self.noi) for i in range(self.noi)]
        self.name = 'Player_' + str(player_num + 1)

    def gather_notes(self):
        self.notes = []
        for i, inst in enumerate(self.insts):
            for lp_note in inst.lp_notes:
                self.notes.append(Note(lp_note, i))
        self.notes.sort(key = lambda x: x.delta)

    def gather_measures(self):
        measure_sizes = pulses_to_measures(self.pulse_sizes)
        self.measures = [Measure(ps, sum(measure_sizes[:i]), self.noi) for i, ps in enumerate(measure_sizes)]
        for measure in self.measures:
            for note in self.notes:
                if note.delta >= measure.pulse_start and note.delta < measure.pulse_end:
                    measure.notes.append(note)

    def notate(self, measures, file_name):
        f = open('saves/lilypond/'+file_name+'.ly', 'w')
        l_text = file_name + ' = {\n'
        l_text += """\clef "percussion" \stopStaff """
        l_text += """\override Staff.StaffSymbol #'line-count = #"""+str(self.noi)+'\n'
        l_text += "\override Staff.StaffSymbol.line-positions = #'("+lp_line_pos(self.noi)+")\n"
        l_text += "\startStaff \key c \major\n"
        prev_pulse_size = ''
        for m, measure in enumerate(measures):
            sig = to_time_sig(measure.pulse_size)
            l_text += '\n%'+str(m+1)+'\n'
            if measure.pulse_size != prev_pulse_size:
                l_text += '\\numericTimeSignature ' + '\\time '+ sig +'\n'
            prev_pulse_size = measure.pulse_size

            l_text += measure.notate()
        l_text += '}'
        f.write(l_text)
        # for measure in measures:
        f.close()





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
        self.quantize()
        self.print_q_midi()
        self.make_lp_ready_notes()
        self.pop_pulse_sizes_to_players()
        self.notate()
        print()


    def notate(self):
        for player in self.players:
            player.gather_notes()
            player.gather_measures()
            for measure in player.measures:
                measure.remove_duplicates()
                measure.chordify()
                measure.pulsify()
            player.notate(player.measures, player.name)

    def pop_pulse_sizes_to_players(self):
        for player in self.players:
            player.pulse_sizes = self.pulse_sizes

    def make_lp_ready_notes(self):
        pulse_collection = Pulse_Collection(100, self.pulse_sizes)
        for inst in self.insts:
            inst.lp_notes = []
            for n, q_note in enumerate(inst.q_notes):
                delta = q_note[1]
                for index in range(len(pulse_collection.pc)):
                    if pulse_collection.pc[index].start_time <= delta and pulse_collection.pc[index].end_time > delta:
                        pulse = pulse_collection.pc[index]
                        pulse_num = index
                        loc_in_pulse = (delta - pulse.start_time) / pulse.dur_tot
                        pulse_loc = pulse_num + loc_in_pulse
                pulse_loc = pulse_loc
                lp_note = [pulse_loc, ['pp', 'p', 'mp', 'mf'][[20, 50, 80, 110].index(q_note[3])]]
                inst.lp_notes += [lp_note]

    def quantize(self):
        self.pulse_sizes = iter_pc(100, self.all_locs).pulse_sizes
        for player in self.players:
            player.locs = []
            for inst in player.insts:
                player.locs += [i[1] for i in inst.notes]
            pc = Pulse_Collection(100, self.pulse_sizes)
            pc.quantize(player.locs)
            player.subdivs = pc.chosen_subdivs
            # pulse_sizes = iter_pc(100, player.locs).pulse_sizes
            for inst in player.insts:
                inst.locs = [i[1] for i in inst.notes]
                pc = Pulse_Collection(100, self.pulse_sizes)
                for p, pulse in enumerate(pc.pc):
                    # print(self.subdivs[p])
                    pulse.subdivs = [player.subdivs[p]]
                    pulse.make_grids()
                pc.quantize(inst.locs)
                inst.q_locs = pc.q_locs
                inst.q_notes=[]
                for n, note in enumerate(inst.notes):
                    # print()
                    # print(inst.locs)
                    # print(inst.q_locs)
                    q_note = [note[0], inst.q_locs[n], note[2], note[3]]
                    inst.q_notes.append(q_note)

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
        dist = ndm(self.noi - 1) ** 0.1
        if sum(dist) == 0: print('dist is where the problem is')
        dist = normalize(dist)
        # dist = normalize(ndm(self.noi) ** 0.1)
        for section in self.sections:
            section.noi = np.random.choice(np.arange(self.noi - 1)+1, p=dist)
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
        self.all_locs = [i[1] for i in notes]
        easy_midi_generator(notes, path + 'all_together.mid', 'Trumpet')

    def print_q_midi(self):
        path = 'saves/midi/'
        notes = [inst.q_notes for inst in self.insts]
        notes = [i for i in itertools.chain.from_iterable(notes)]

        easy_midi_generator(notes, path + 'q_all_together.mid', 'Trumpet')
