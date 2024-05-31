import numpy as np

from craft import fdmt as FDMT
import sys
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/Baraks_FDMT/")
import FDMT_no_correction as BZ_FDMT
import numba
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/")
import simulate_narrow_frb as snf
import argparse

def nearest_largest_power_of_2(n):
    power_of_2s = 2**np.arange(0, 30)
    useful_power_of_2s = power_of_2s[power_of_2s > n]
    return useful_power_of_2s[0]

def main():
    dm_trials = np.arange(args.dm_min, args.dm_max, args.dm_step)
    nsamps = int(nearest_largest_power_of_2(args.dm_max))
    nch = int(args.nch)
    tx = 1000
    f_low_cen = args.f_low_cen
    chw = args.bw / args.nch
    chw_2 = chw / 2
    f_high_cen = f_low_cen + (args.nch - 1) * chw
    fchans = np.linspace(f_low_cen, f_high_cen, nch, endpoint=True, dtype=np.float32)
    
    
    max_dm_searched = args.dm_max + 1
    assert max_dm_searched < nsamps - 10, "We need to have some buffer between the max DM searched and the no of samps in the block"
    tpulse = nsamps - 10 + args.spp

    thefdmt = FDMT.Fdmt(f_min = fchans[0], f_off = chw, n_f = nch, max_dt = max_dm_searched, n_t = nsamps)

    if args.outname is None:
        outname = f"noc_fdmt_performance_nch_{nch}_bw_{args.bw}_{args.dm_min}_to_{args.dm_max}_step_{args.dm_step}_with_bz_spp_{args.spp}.txt"
    else:
        outname = args.outname

    #f = open("fdmt_performance_nch_128_0_to_4000_with_bz_spp0.5.txt", 'w')
    f = open(outname, 'w')
    f.write("DM\tMax_snr\tFDMT_dm\tFDMT_snr\tBZ_FDMT_dm\tBZ_FDMT_snr\n")
    ones = np.ones((nch, nsamps), dtype=np.float32)
    bz_fdmt_ones = BZ_FDMT.FDMT(ones,  fchans[0] - chw_2, fchans[-1] + chw_2, max_dm_searched, np.float32)
    for idm in dm_trials:
        x, tot_samps = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
        max_snr = np.sum(x) / np.sqrt(tot_samps)
        
        bz_fdmt = BZ_FDMT.FDMT(x, fchans[0] - chw_2, fchans[-1] + chw_2, max_dm_searched, np.float32)
        bz_fdmt_peak_dm, bz_fdmt_peak_time = np.unravel_index(np.argmax(bz_fdmt), bz_fdmt.shape)
        bz_fdmt_samps_added = bz_fdmt_ones[bz_fdmt_peak_dm, bz_fdmt_peak_time]
        bz_fdmt_snr = np.max(bz_fdmt) / np.sqrt(bz_fdmt_samps_added)
        
        dx = thefdmt(x)
        fdmt_peak_dm, _ = np.unravel_index(np.argmax(dx), dx.shape)
        fdmt_snr = np.max(dx) / thefdmt.get_eff_sigma(fdmt_peak_dm, 1)

        out_str = f"{idm:.2f}\t{max_snr:.2f}\t{fdmt_peak_dm}\t{fdmt_snr:.2f}\t{bz_fdmt_peak_dm:.2f}\t{bz_fdmt_snr:.2f}\n"
        f.write(out_str)
        print(idm)
    f.close()


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('-dm_min', type=float, help="Min DM (def = 0)", default=0)
    a.add_argument('-dm_max', type=float, help="Max DM (def = 1000)", default=1000)
    a.add_argument('-dm_step', type=float, help="Min DM (def = 0.1)", default=0.1)

    a.add_argument('-nch', type=int, help="No of channels (default=256)", default=256)
    a.add_argument('-bw', type=float, help="BW in MHz (default=256)", default=256)
    a.add_argument('-f_low_cen', type=float, help="Center freq of lowest channel (default=800.5)", default=800.5)

    a.add_argument('-spp', type=float, help="Sub-sample phase of the pulse TOA (def = 0.5)", default=0.5)

    a.add_argument("-outname", type=str, help="Name of the output file (def=fdmt_performance_nch_{}_bw_{}_{dm_min}_to_{dm_max}_step_{dm_step}_with_bz_spp_{spp}.txt)", default=None)

    args = a.parse_args()

    main()