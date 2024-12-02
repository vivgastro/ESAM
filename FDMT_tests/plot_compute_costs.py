import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    plt.figure()
    for ii, fname in enumerate(args.pc_files):
        counts = np.loadtxt(fname, delimiter=" ")
        #pc_growth = counts[counts[:, 0] > 0].sum(axis=1)
        pc_growth = counts.sum(axis=1)
        plt.plot(pc_growth, '.', label=fname)

    plt.yscale('log')
    plt.xlabel("Number of DMs/templates")
    plt.ylabel("No of operations")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("pc_files", nargs = '+', type=str, help="Path to the product count files")
    args = a.parse_args()
    main()
