import numpy as np
import random
import itertools
from inspect import signature
import pretty_midi
import os

def normalize(array):
    array = np.array(array)
    return array / sum(array)

def get_partition(n):
    """Gets the partition"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
            x = a[(k - 1)] + 1
            k -= 1
            while 2 * x <= y:
                    a[k] = x
                    y -= x
                    k += 1
            l = k + 1
            while x <= y:
                    a[k] = x
                    a[l] = y
                    yield a[:k + 2]
                    x += 1
                    y -= 1
            a[k] = x + y
            y = x + y - 1
            yield a[:k + 1]

def select_partition(n, type = 'equal', one_allowed='yes'):
    partition = list(get_partition(n))
    if one_allowed == 'no':
        partition = partition[:-1]
    if np.shape(partition) == (1,1): return partition[0]
    if type == 'equal':
        # print(np.shape(partition))
        return np.random.choice(partition)
    elif type == 'rounded':
        p = normalize([1 / len(i) for i in partition])
        return np.random.choice(partition, p = p)
    # choice = np.random.randint(0, len(p))
    # return p[choice]


def dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0):
    selections = []
    if np.all(counts == 0):
        counts = [
         1] * len(choices)
    weights = np.array(weights)
    if np.all(weights) == 0:
        weights = [
         1] * len(choices)
    for q in range(epochs):
        sum_ = sum([weights[i] * counts[i] ** alpha for i in range(len(choices))])
        probs = [weights[i] * counts[i] ** alpha / sum_ for i in range(len(choices))]
        selection_index = np.random.choice((list(range(len(choices)))), p=probs)
        counts = [i + 1 for i in counts]
        counts[selection_index] = 0
        selections.append(choices[selection_index])

    selections = np.array(selections)
    counts = np.array(counts)
    if verbosity == 0:
        return selections
    if verbosity == 1:
        return (
         selections, counts)

def normal_distribution_maker(bins):
    distribution = np.random.normal(size=100000)
    distribution = np.histogram(distribution, bins=bins, density=True)[0]
    distribution /= np.sum(distribution)
    return distribution

def nPVI(d):
    m = len(d)
    return 100 / (m - 1) * sum([abs((d[i] - d[(i + 1)]) / (d[i] + d[(i + 1)]) / 2) for i in range(m - 1)])

def nPVI_averager(window_width, durs):
    return [nPVI(durs[i:i + window_width]) for i in range(len(durs) - window_width)]

def nCVI(d):
    matrix = [list(i) for i in itertools.combinations(d, 2)]
    matrix = [nPVI(i) for i in matrix]
    return sum(matrix) / len(matrix)

def segment(num_of_segments,nCVI_average,factor=2.0):
    section_durs = factor ** np.random.normal(size=2)
    while abs(nCVI(section_durs) - nCVI_average) > 1.0:
        section_durs = factor ** np.random.normal(size=2)
    for i in range(num_of_segments - 2):
        next_section_durs = np.append(section_durs,[factor ** np.random.normal()])
        ct=0
        while abs(nCVI(next_section_durs) - nCVI_average) > 1.0:
            ct+=1
            next_section_durs = np.append(section_durs, [factor ** np.random.normal()])
        section_durs = next_section_durs
        # print(ct)
    section_durs /= np.sum(section_durs)
    return section_durs

def auto_args(target):
    """
    A decorator for automatically copying constructor arguments to `self`.
    """
    # Get a signature object for the target method:
    sig = signature(target)
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        target(self, *args, **kwargs)
    return replacement

def spread(init, max_ratio):
    exponent = np.clip(np.random.normal() / 3, -1, 1)
    return init * (max_ratio ** exponent)

def dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0):
    selections = []
    if np.all(counts == 0):
        counts = [
         1] * len(choices)
    weights = np.array(weights)
    if np.all(weights) == 0:
        weights = [
         1] * len(choices)
    for q in range(epochs):
        sum_ = sum([weights[i] * counts[i] ** alpha for i in range(len(choices))])
        probs = [weights[i] * counts[i] ** alpha / sum_ for i in range(len(choices))]
        selection_index = np.random.choice((list(range(len(choices)))), p=probs)
        counts = [i + 1 for i in counts]
        counts[selection_index] = 0
        selections.append(choices[selection_index])
    selections = np.array(selections)
    counts = np.array(counts)
    if verbosity == 0:
        return selections
    if verbosity == 1:
        return (
         selections, counts)

def dc_weight_finder(choices, alpha, weights, test_epochs=500):
    choices = np.arange(len(choices))
    weights_ = [i / sum(weights) for i in weights]
    max_off = .051
    # cts_ = 0
    test_ct = 0
    while max_off > 0.05:
        test_ct += 1
        if (test_ct > 1000) and (test_ct%100 == 0): print(test_ct)

        # print(cts_)
        y = dc_alg(choices, test_epochs, alpha, weights)
        #this should be rewritten as a np function
        results = np.array([np.count_nonzero(y==choices[i]) / test_epochs for i in choices])
        results = np.where(results == 0, results, 0.001)
        diff = weights_ / results
        weights *= diff
        weights /= sum(weights)
        max_off = np.max(1 - diff)
        # print(max_off)
        # cts_+=1
        # print(cts_)
    return weights

def weighted_dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0, weights_dict={}):
    if np.any(weights) != 0:
        # this basically says if its not going to work, just double the length
        #of the choice array, and try again. Might be better to just double the
        # one value thats above 0.5 . Or, might make more sense to just do a straight
        # random choice.
        if np.max(weights) >= 0.5:
            choices = np.tile(choices, 2)
            weights = np.tile(weights/2, 2)
            counts = np.tile(counts, 2)

        #if there are any weights that are 0, this will just remove that item
        #from the weights and from the choices, so dc_weight_finder doesn't break
        nonzero_locs = np.nonzero(weights != 0)
        choices = choices[nonzero_locs]
        weights = weights[nonzero_locs]
        weights = dc_weight_finder(choices, alpha, weights)
    selections = dc_alg(choices, epochs, alpha, weights, counts, verbosity)
    return selections


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd='\r'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "
", "
") (Str)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)), end=printEnd)
    if iteration == total:
        print()

def easy_midi_generator(notes, file_name, midi_inst_name):
    notes = sorted(notes, key=(lambda x: x[1]))
    score = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(midi_inst_name)
    instrument = pretty_midi.Instrument(program=0)
    for n, note in enumerate(notes):
        if type(note[3]) == np.float64:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == float:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == int:
            vel = note[3]
        else: print(note[3])
        note = pretty_midi.Note(velocity=vel, pitch=(note[0]), start=(note[1]), end=(note[1] + note[2]))
        instrument.notes.append(note)
    score.instruments.append(instrument)
    score.write(file_name)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #returns correct item and error
    return array[idx], np.abs(array[idx] - value)

def quantize(locs, grid):
    new_locs = []
    error = 0
    for loc in locs:
        new_loc, err = find_nearest(grid, loc)
        new_locs.append(new_loc)
        error += err
    return new_locs, error

def split_at_flip_points(pulse_list):
    pl = pulse_list
    havs = [i for i in range(len(pl)) if pl[i] == 0.5]
    subgroups =[]
    for i, item in enumerate(havs):
        if i == 0:
            group = pl[:item+1]
        else:
            group = pl[havs[i-1]+1:item+1]
            subgroups.append(group)
    return subgroups

def measure_up(grp):
    out = []
    while len(grp) > 10:
        out += [4]
        grp = grp[4:]
    if len(grp) == 1:
        print('No No No!')
    elif len(grp) == 2:
        # 3/8
        out += [1.5]
    elif len(grp) == 3:
        # 5/8
        out += [2.5]
    elif len(grp) == 4:
        # 7/8
        out += [3.5]
    elif len(grp) == 5:
        # 2/4, 5/8
        out += [2, 2.5]
    elif len(grp) == 6:
        # 2/4, 7/8
        out += [2, 3.5]
    elif len(grp) == 7:
        # 3/4, 7/8
        out += [3, 3.5]
    elif len(grp) == 8:
        # 4/4, 7/8
        out += [4, 3.5]
    elif len(grp) == 9:
        # 3/4, 3/4, 5/8
        out += [3, 3, 2.5]
    elif len(grp) == 10:
        # 3/4, 3/4, 7/8
        out += [3, 3, 3.5]
    return out

def pulses_to_measures(pulse_sizes):
    if np.all(np.array(pulse_sizes) == 1):
        return [5 for i in range(np.int(np.ceil(len(pulse_sizes)/5)))]
    subgroups = split_at_flip_points(pulse_sizes)
    measures = [measure_up(group) for group in subgroups]
    return [i for i in itertools.chain.from_iterable(measures)]

def run_lily(fn, dir = 'saves/lilypond'):
    os.system('docker run --rm -v $(pwd):/app -w /app gpit2286/lilypond lilypond '+ dir +'/'+fn+'.ly')
    os.rename(fn+'.pdf', dir+'/'+fn+'.pdf')

def bpm_to_pulse_dur(bpm):
    return 60 / bpm

def delta_to_pulse_loc(delta, bpm):
    pulse_dur = bpm_to_pulse_dur(bpm)
    return delta/pulse_dur

def to_time_sig(pulse_size):
    if pulse_size == 5: sig = "5/4"
    if pulse_size == 4: sig = "4/4"
    if pulse_size == 3.5: sig = "7/8"
    if pulse_size == 3: sig = "3/4"
    if pulse_size == 2.5: sig = "5/8"
    if pulse_size == 2: sig = "2/4"
    if pulse_size == 1.5: sig = "3/8"
    return sig

def lp_line_pos(num):
    if num == 6: out = "-7 -4 -1 2 5 8"
    if num == 5: out = "-4 -2 0 2 4"
    if num == 4: out = "-3 -1 1 3"
    if num == 3: out = "-2 0 2"
    if num == 2: out = "-1 1"
    return out




def number_to_english(n):
    TENS = {30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}
    ZERO_TO_TWENTY = (
        'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
        'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty'
    )
    if any(not x.isdigit() for x in str(n)):
        return ''

    if n <= 20:
        return ZERO_TO_TWENTY[n]
    elif n < 100 and n % 10 == 0:
        return TENS[n]
    elif n < 100:
        return number_to_english(n - (n % 10)) + ' ' + number_to_english(n % 10)
    elif n < 1000 and n % 100 == 0:
        return number_to_english(n / 100) + ' hundred'
    elif n < 1000:
        return number_to_english(n / 100) + ' hundred ' + number_to_english(n % 100)
    elif n < 1000000:
        return number_to_english(n / 1000) + ' thousand ' + number_to_english(n % 1000)

    return ''
