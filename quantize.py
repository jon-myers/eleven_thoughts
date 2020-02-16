import numpy as np
from funcs import quantize, print_progress_bar, bpm_to_pulse_dur
from matplotlib import pyplot as plt
import itertools



class Pulse:
    #subdives aught to be co-prime
    def __init__(self, start_time, bpm, subdivs=[3, 4, 5]):
        self.subdivs = subdivs
        self.start_time = start_time
        self.dur_tot = bpm_to_pulse_dur(bpm)
        self.end_time = self.start_time + self.dur_tot
        self.make_grids()


    def make_grids(self):
        self.grids = [[self.start_time + (i * self.dur_tot / div) for i in range(div+1)] for div in self.subdivs]

    def choose_quantization(self, locs):
        self.err = len(locs) * 10 + 1000
        for i, grid in enumerate(self.grids):
            new_locs, error = quantize(locs, grid)
            if i == 2:
                error *= 1.6
            if error < self.err:
                self.locs = new_locs
                self.err = error
                self.chosen_subdiv = self.subdivs[i]
            # elif error == 0:
            #     self.err = 0:

        # print(self.chosen_subdiv)

class Half_Pulse:
    #div gotta be 2
    def __init__(self, start_time, bpm, subdivs=[2]):
        self.subdivs = subdivs
        self.start_time = start_time
        self.dur_tot = bpm_to_pulse_dur(bpm) / 2
        self.end_time = self.start_time + self.dur_tot
        self.grids = self.make_grids()

    def make_grids(self):
        return [[self.start_time + (i * self.dur_tot / div) for i in range(div+1)] for div in self.subdivs]

    def choose_quantization(self, locs):
        # self.err = len(locs)
        # for i, grid in enumerate(self.grids):
        #     new_locs, error = quantize(locs, grid)
        #     if error < self.err:
        #         self.locs = new_locs
        #         self.err = error
        # print()
        # print('grids comin up')
        # print(self.grids)
        # print(self.grids[0])
        self.locs, self.err = quantize(locs, self.grids[0])
        self.chosen_subdiv = 2

class Pulse_Collection:
    def __init__(self, bpm, pulse_sizes, st=0):
        self.st = st
        self.bpm = bpm
        self.pulse_sizes = pulse_sizes
        self.obj_dict = {1: Pulse, 0.5: Half_Pulse}
        self.pc=[]
        self.add_to_collection()

    def add_to_collection(self):
        st=self.st
        for ps in self.pulse_sizes:
            pulse_type = self.obj_dict[ps]
            pulse = pulse_type(st, self.bpm)
            self.pc.append(pulse)
            st+=pulse.dur_tot

    def quantize(self, locs):
        carried_err = 0
        self.chosen_subdivs=[]
        for pulse in self.pc:
            relevent_locs = [i for i in locs if i>=pulse.start_time and i<pulse.end_time]
            pulse.choose_quantization(relevent_locs)
            carried_err += pulse.err
            self.chosen_subdivs.append(pulse.chosen_subdiv)
        self.err = carried_err
        self.q_locs = [pulse.locs for pulse in self.pc]
        self.q_locs = [i for i in itertools.chain.from_iterable(self.q_locs)]



def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b

def best_fit_func_gen(x, y):
    a, b = best_fit(x, y)
    z = np.polyfit(x, y, 3)
    return np.poly1d(z)

# iteratively build Pulse_Collection
def iter_pc(bpm, locs, look_ahead = 16):
    pulse_dur = bpm_to_pulse_dur(bpm)
    end_time = np.max(locs)
    tot_pulses = np.ceil(end_time / pulse_dur)
    st=0
    pulse_sizes = []
    window_dur = (look_ahead + 0.5) * pulse_dur
    print_progress_bar(0, end_time - window_dur, 'Progress: ', length = 50)
    ticker = 0
    while st < end_time - window_dur:
        relevant_locs = [i for i in locs if i >= st and i < st + window_dur]
        a = Pulse_Collection(bpm, [1 for i in range(look_ahead)] + [0.5], st)
        b = Pulse_Collection(bpm, [0.5] + [1 for i in range(look_ahead)], st)
        a.quantize(relevant_locs)
        b.quantize(relevant_locs)
        if a.err == 0:
            pulse_sizes.append(1)
            st += (1 * pulse_dur)
        elif b.err / a.err > 1.05:
            pulse_sizes.append(0.5)
            st += (0.5 * pulse_dur)
        else:
            pulse_sizes.append(1)
            st += (1 * pulse_dur)
        print_progress_bar(st, end_time - window_dur, 'Progress: ', length = 50)
    pulse_sizes = pulse_sizes + [1 for i in range(look_ahead)] + [0.5]
    fin_pc = Pulse_Collection(bpm, pulse_sizes, 0)
    fin_pc.quantize(locs)
    return fin_pc

# import pickle
# from compose import Piece
# piece = pickle.load(open('saves/pickles/piece.p', 'rb'))
# test_locs = piece.all_locs
# idea_2 = iter_pc(100, test_locs, look_ahead=16)
# print(idea_2.pulse_sizes)

# pickle.dump(idea_2.pulse_sizes, open('saves/pickles/pulse_sizes.p', 'wb'))
