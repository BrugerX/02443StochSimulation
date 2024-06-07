from RNG.RNG import LCG,UniformDistribution
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sstats

# task b.3) - Chi2 test
def chiTestStat(nrIntervals, samples):
    nrSamples = len(samples)
    expectedPerInterval = nrSamples / nrIntervals
    intervals = [x * (1 / nrIntervals) for x in range(nrIntervals + 1)]
    realPerInterval = np.zeros(nrIntervals)
    np_samples = np.array(samples)
    chi2 = sstats.chi2(df=nrIntervals - 1)

    for i in range(len(intervals) - 1):
        a = intervals[i]
        b = intervals[i + 1]
        realPerInterval[i] = ((a <= np_samples) & (np_samples < b)).sum()

    test_stat = sum([(nrObserved - expectedPerInterval) ** 2 / expectedPerInterval for nrObserved in realPerInterval])

    p_value = 1 - chi2.cdf(test_stat)
    return p_value


# task b.4) - Kolmogorov Smirnov
def Fe(x, sample):
    return (np.array(sample) <= x).sum() / len(sample)


def ksmir(samples):
    nrSamples = len(samples)
    differencesF = [np.abs(Fe(x, samples) - x) for x in samples]
    Dn = (np.sqrt(nrSamples) + 0.12 + 0.11 / np.sqrt(nrSamples)) * np.max(differencesF)
    return Dn


# task b.6) Correlation

def corrTest(h, samples):
    nrSamples = len(samples)
    productSum = 0
    for i in range(nrSamples - h):
        productSum += samples[i] * samples[i + h]

    test_stat_corr = productSum * 1 / (nrSamples - h)

    nDist = sstats.norm(0.25, np.sqrt(7 / (144 * nrSamples)))

    p_value_corr = nDist.cdf(test_stat_corr)

    if (test_stat_corr >= 0.25):
        p_value_corr = 1 - p_value_corr

    return p_value_corr


def count_runs(array):
    seq_of_runs = []
    prev = -np.inf
    cur_run_length = 0
    for i in range(len(array)):
        if array[i] > prev:
            cur_run_length += 1
            prev = array[i]
        elif array[i] < prev:
            seq_of_runs.append(cur_run_length)
            cur_run_length = 1
            prev = array[i]

    return np.array(seq_of_runs)


def up_down_knuth(generated_numbers):
    R = np.zeros(6)
    n = len(generated_numbers)
    seq_runs = count_runs(generated_numbers)
    R[0:5] = [sum(seq_runs == x) for x in range(1, 6)]
    R[-1] = sum(seq_runs >= 6)

    A = np.array([[4529.4, 9044.9, 13568, 18091, 22615, 27892], [9044.9, 18097, 27139, 36187, 45234, 55789],
                  [13568, 27139, 40721, 54281, 67852, 83685], [18091, 36187, 54281, 72414, 90470, 111580],
                  [22615, 45234, 67852, 90470, 113262, 139476], [27892, 55789, 83685, 111580, 139476, 172860]])

    B = np.array([1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840])

    chi2_stat = ((R - n * B).T @ A @ (R - n * B)) / (n - 6)

    return 1 - sstats.chi2.cdf(chi2_stat, 6)  # return p-value