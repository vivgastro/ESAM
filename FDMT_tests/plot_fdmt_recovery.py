import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def main():
    f = plt.figure()
    ax = f.add_subplot(111)
    for ii, fname in enumerate(args.fname):
        if len(args.labels) > 0:
            label = args.labels[ii]
        else:
            label = fname
        results = pd.read_csv(fname, sep='\s+')
        ax.plot(results['DM'], results['BZ_FDMT_snr'] / results['MF_snr'], label=label)
    
    plt.axhline(1, ls = '--', c='k')
    plt.legend()
    plt.xlabel("DM [samples]")
    plt.ylabel("Fraction of S/N recovered")
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("fname", type=str, nargs = '+',  help="Path to the FDMT performance results file")
    a.add_argument("-labels", type=str, nargs = '+', help="Labels")
    args = a.parse_args()
    main()
