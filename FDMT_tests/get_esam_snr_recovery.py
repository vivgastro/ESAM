import sys
sys.path.append("/home/gup037/Codes/ESAM/esam")
import esam, traces

sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests")
import simulate_narrow_frb as snf


import numpy as np


def main():
    nch = 256
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    chw_2 = 0.5
    nsamps = 4096
    tpulse = 4018.5
    tx = 1000

    dm_linear_tests = np.arange(0, 4000, 50)
    dm_random_tests = np.random.uniform(0, 4000, 10)
    dm_close_to_1_tests = np.random.exponential(scale=2, size=10)

    dm_tests = np.concatenate([dm_linear_tests, dm_random_tests, dm_close_to_1_tests])

    #load the tree
    tree = np.load("esam_tree_dm_0_1000_0.1_spp0.5_nch256.pkl.npy", allow_pickle=True).item()
    tree = np.load("esam_tree_fdmt_tracks_0_1000_1_nch256.pkl.npy", allow_pickle=True).item()
    tree = np.load("esam_tree_fdmt_tracks_0_4000_1_nch256.pkl.npy", allow_pickle=True).item()

    #now use the tree to get snrs

    f = open("esam_snr_recovery_fdmt_tracks_only_0_4000.txt", 'w')
    f.write("DM\tMax_snr\tESAM_dm\tESAM_snr\n")
    #first get the rms values by processing a block of ones
    ones_dmout = tree(np.ones((nch, nsamps)))
    for ii, idm in enumerate(dm_tests):
        x, tot_samps_added = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
        max_snr = np.sum(x) / np.sqrt(tot_samps_added)
        dmout = tree(x)
        peak_dm_loc, peak_time_loc = np.unravel_index(np.argmax(dmout), dmout.shape)
        peak_val = dmout[peak_dm_loc, peak_time_loc]
        peak_snr = peak_val / np.sqrt(ones_dmout[peak_dm_loc, peak_time_loc])

        out_str = f"{idm}\t{max_snr}\t{peak_dm_loc}\t{peak_snr}\n"
        f.write(out_str)
        print(idm)
    f.close()


main()
