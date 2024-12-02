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
    a.add_argument("-nch", type=int, help="Nchan (nch:256)", default=256)
    a.add_argument("-dmmax", type=int, help="Maximim DM upto which we should test (def:1000)", default=1000)
    a.add_argument("-dmstep", type=float, help="Steps in which to increment the DM trials (def:0.1)", default=0.1)
    a.add_argument("-bf_dm_trial_spacing", type=float, help="The DM trials which the BF algo will remember to execute (similar to the spacing of templates given to ESAM) (def: 1)", default=1.0)

    args = a.parse_args()
    return args

def my_convolve(a, b):
    '''
    a is the kernel
    b is the data
    '''
    assert a.shape[0] == b.shape[0]
    if a.shape[1] > b.shape[1]:
        a = a[:, -b.shape[1]:]
    norm = np.sqrt((a**2).sum())
    off_trials = np.arange(-2, 3)
    out = np.zeros(len(off_trials))
    for ii, delay in enumerate(off_trials):
        out[ii] = np.sum(np.roll(a, delay, axis=1) * b) / norm

    return out

def main():

    args = get_parser()
    nch = args.nch
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


    outname = f"final_bf_snr_recovery_fine_dm_grid_dm_0_{args.dmmax}_{args.dmstep}_using_nch={nch}.txt"
    f = open(outname, 'w')
    f.write("DM\tMF_snr\tMax_snr\tBF_dm\tBF_snr\n")
    #first get the rms values by processing a block of ones

    print("Finished getting the ones_dmout")
    for ii, idm in enumerate(dm_tests):
        x, tot_samps_added = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)

        if idm % args.bf_dm_trial_spacing == 0:
            x_pre = np.zeros_like(x)
            x_pre[x>0] = 1
            tmp , _ = snf.make_pure_frb(nsamps, nch, tx, idm+1, fchans, chw_2, tpulse)
            x_post = np.zeros_like(x)
            x_post[tmp > 0] = 1
            
        max_snr = np.sum(x) / np.sqrt(tot_samps_added)
        mf_snr = np.sqrt(np.sum(x**2))

        best_pre_snr = np.max(my_convolve(x_pre, x))
        best_post_snr = np.max(my_convolve(x_post, x))
        if best_pre_snr > best_post_snr:
            bf_snr = best_pre_snr
            bf_dm = int(idm)
        else:
            bf_snr = best_post_snr
            bf_dm = int(idm) + 1

        out_str = f"{idm}\t{mf_snr}\t{max_snr}\t{bf_dm}\t{bf_snr}\n"
        f.write(out_str)
        print(idm)
        if ii % 100 == 0:
            f.flush()
    f.close()


main()
