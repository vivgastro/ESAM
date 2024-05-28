import numpy as np

from craft import fdmt as FDMT
import sys
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/Baraks_FDMT/")
import FDMT as BZ_FDMT
import numba
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/")
import simulate_narrow_frb as snf


dm_trials = np.arange(0, 1000, 0.1)
nsamps = 4096
nch = 128
tx = 1000
f_low_cen = 800.5
f_high_cen = 1055.5 
fchans = np.linspace(f_low_cen, f_high_cen, nch, endpoint=True, dtype=np.float32)
chw_2 = (fchans[-1] - fchans[0]) / (nch -1)
chw = chw_2 * 2
tpulse = 4095.5
max_dm = 4090

thefdmt = FDMT.Fdmt(f_min = fchans[0], f_off = chw_2 * 2, n_f = nch, max_dt = max_dm, n_t = nsamps)

f = open("fdmt_performance_nch_128_0_to_4000_with_bz_spp0.5.txt", 'w')
f.write("DM\tMax_snr\tFDMT_dm\tFDMT_snr\tBZ_FDMT_dm\tBZ_FDMT_snr\n")
ones = np.ones((nch, nsamps), dtype=np.float32)
bz_fdmt_ones = BZ_FDMT.FDMT(ones,  fchans[0] - chw_2, fchans[-1] + chw_2, max_dm, np.float32)
for idm in dm_trials:
    x, tot_samps = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
    max_snr = np.sum(x) / np.sqrt(tot_samps)
    
    bz_fdmt = BZ_FDMT.FDMT(x, fchans[0] - chw_2, fchans[-1] + chw_2, max_dm, np.float32)
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
