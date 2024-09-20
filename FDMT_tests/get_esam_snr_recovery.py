import sys
sys.path.append("/home/gup037/Codes/ESAM/esam")
import esam, traces

sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests")
import simulate_narrow_frb as snf


import numpy as np

def nearest_largest_power_of_2(n):
    power_of_2s = 2**np.arange(0, 30)
    useful_power_of_2s = power_of_2s[power_of_2s > n]
    return useful_power_of_2s[0]


def main():
    nch = 256
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    chw_2 = 0.5
    max_dm = 100
    nsamps = nearest_largest_power_of_2(max_dm)
    tpulse = nsamps - 20.5
    tx = 1000

    dm_linear_tests = np.arange(0, max_dm, 10)
    dm_random_tests = np.random.uniform(0, max_dm, 100)
    dm_close_to_1_tests = np.random.exponential(scale=2, size=10)

    dm_tests = np.concatenate([dm_linear_tests, dm_random_tests, dm_close_to_1_tests])
    dm_tests = np.arange(0, max_dm, 1)
    #load the tree
    
    #tree_fname = "esam_tree_dm_0_1000_0.1_spp0.5_nch256.pkl.npy"
    tree_fname = "esam_tree_fdmt_tracks_0_1000_1_nch256.pkl.npy"
    #tree_fname = "esam_tree_fdmt_tracks_0_4000_1_nch256.pkl.npy"
    tree_fname = "esam_tree_dm_0_1000_1_spp0.5_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_40_1_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_1000_100_nch256.pkl.npy"
    tree_fname = "esam_tree_with_match_filter_traces_0_1000_10_nch256.pkl.npy"
    tree_fname = "optimised_esam_tree_with_match_filter_traces_0_100_0.1_nch256_threshold_0.9.pkl.npy"

    tree = np.load(tree_fname, allow_pickle=True).item()
    #now use the tree to get snrs
    print("Finished loading the pickled tree")

    outname = f"esam_snr_recovery_fine_dm_grid_using_{tree_fname}.txt"
    f = open(outname, 'w')
    f.write("DM\tMF_snr\tMax_snr\tESAM_dm\tESAM_snr\n")
    #first get the rms values by processing a block of ones
    ones_dmout = tree(np.ones((nch, nsamps)), squared_weights = True)

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
