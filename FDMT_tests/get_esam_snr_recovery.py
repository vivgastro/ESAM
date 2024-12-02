import sys, os
sys.path.append("/home/gup037/Codes/ESAM/esam")
import esam, traces
import matplotlib.pyplot as plt
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests")
import simulate_narrow_frb as snf
import argparse

import numpy as np

def nearest_largest_power_of_2(n):
    power_of_2s = 2**np.arange(0, 30)
    useful_power_of_2s = power_of_2s[power_of_2s > n]
    return useful_power_of_2s[0]

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-tree", type=str, help="Path to the tree file to load and test", required=True)
    a.add_argument("-dmmax", type=int, help="Maximim DM upto which we should test (def:1000)", default=1000)
    a.add_argument("-dmstep", type=float, help="Steps in which to increment the DM trials (def:0.1)", default=1)

    args = a.parse_args()
    return args

def main():

    args = get_parser()
    #load the tree
    
    #tree_fname = "esam_tree_dm_0_1000_0.1_spp0.5_nch256.pkl.npy"
    tree_fname = "esam_tree_fdmt_tracks_0_1000_1_nch256.pkl.npy"
    #tree_fname = "esam_tree_fdmt_tracks_0_4000_1_nch256.pkl.npy"
    tree_fname = "esam_tree_dm_0_1000_1_spp0.5_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_40_1_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_1000_100_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_1000_10_nch256.pkl.npy"
    tree_fname = "optimised_esam_tree_with_match_filter_traces_0_100_0.1_nch256_threshold_0.9.pkl.npy"

    tree_fname = args.tree

    tree = np.load(tree_fname, allow_pickle=True).item()
    
    
    nch = tree.nchan
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    chw_2 = 0.5
    max_dm = args.dmmax
    nsamps = nearest_largest_power_of_2(max_dm)
    tpulse = nsamps - 20.5
    tx = 1000

    #dm_linear_tests = np.arange(0, max_dm, 10)
    #dm_random_tests = np.random.uniform(0, max_dm, 100)
    #dm_close_to_1_tests = np.random.exponential(scale=2, size=10)

    #dm_tests = np.concatenate([dm_linear_tests, dm_random_tests, dm_close_to_1_tests])
    dm_tests = np.arange(0, max_dm, args.dmstep)

    #now use the tree to get snrs
    print("Finished loading the pickled tree")

    outname = f"final_esam_snr_recovery_fine_dm_grid_dm_0_{args.dmmax}_{args.dmstep}_using_{tree_fname}.txt"
    f = open(outname, 'w')
    f.write("DM\tMF_snr\tMax_snr\tESAM_dm\tESAM_snr\n")
    #first get the rms values by processing a block of ones
    ones_dmout = tree(np.ones((nch, args.dmmax + 128)))

    print("Finished getting the ones_dmout")
    for ii, idm in enumerate(dm_tests):
        x, tot_samps_added = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
        max_snr = np.sum(x) / np.sqrt(tot_samps_added)
        mf_snr = np.sqrt(np.sum(x**2))
        dmout = tree(x)
        peak_dm_loc, peak_time_loc = np.unravel_index(np.argmax(dmout), dmout.shape)
        peak_val = dmout[peak_dm_loc, peak_time_loc]
        peak_snr = peak_val / np.sqrt(ones_dmout[peak_dm_loc, peak_time_loc])

        out_str = f"{idm}\t{mf_snr}\t{max_snr}\t{peak_dm_loc}\t{peak_snr}\n"
        f.write(out_str)
        print(idm)
    f.close()


main()
