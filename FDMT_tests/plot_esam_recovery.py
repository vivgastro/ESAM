import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def main():
    f = plt.figure()
    ax = f.add_subplot(111)
    plotting_steps = 1
    for ii, fname in enumerate(args.fname):
        results = pd.read_csv(fname, sep='\s+')
        if len(args.labels) > 0:
            label = args.labels[ii]
        else:
            label = fname
        ax.plot(results['DM'][::plotting_steps], results['ESAM_snr'][::plotting_steps] / results['MF_snr'][::plotting_steps], label=label)
    plt.legend()
    plt.xlabel("DM [samples]")
    plt.ylabel("Fraction of S/N recovered")
    plt.ylim(0.25, 1.05)
    plt.axhline(0.943, ls='--', c='k')
    plt.axhline(0.9, ls='--', c='k')
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("fname", type=str, nargs = '+',  help="Path to the ESAM performance results file")
    a.add_argument("-labels", type=str, nargs='+', help="Labels (comma-separated) one per fname")
    args = a.parse_args()
    main()
