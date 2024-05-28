import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def main():
    f = plt.figure()
    ax = f.add_subplot(111)
    for fname in args.fname:
        results = pd.read_csv(fname, sep='\s+')
        ax.plot(results['DM'], results['BZ_FDMT_snr'] / results['Max_snr'])
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("fname", type=str, nargs = '+',  help="Path to the FDMT performance results file")
    args = a.parse_args()
    main()
