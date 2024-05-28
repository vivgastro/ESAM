import numpy as np

from craft import fdmt as FDMT
import sys
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/Baraks_FDMT/")
import FDMT as BZ_FDMT
import numba
sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests/")
import simulate_narrow_frb as snf


dm_trials = np.arange(0, 1000, 1)
nsamps = 1024
nch = 256
tx = 1000
fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
chw_2 = 0.5
chw = chw_2 * 2
max_dm = 1000
spp_trials = np.arange(0, 1, 0.1)
thefdmt = FDMT.Fdmt(f_min = fchans[0], f_off = chw_2 * 2, n_f = nch, max_dt = max_dm, n_t = nsamps)

f = open("fdmt_performance_0_to_1000_1_with_spp_0_to_0.9_0.1.txt", 'w')
ones = np.ones((nch, nsamps), dtype=np.float32)
#bz_fdmt_ones = BZ_FDMT.FDMT(ones,  fchans[0] - chw_2, fchans[-1] + chw_2, max_dm, np.float32)
for idm in dm_trials:
    f.write(f"{idm}")
    for ispp, spp in enumerate(spp_trials):
         tpulse = 1010 + spp
         
         x, tot_samps = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
         max_snr = np.sum(x) / np.sqrt(tot_samps)
         
         #bz_fdmt = BZ_FDMT.FDMT(x, fchans[0] - chw_2, fchans[-1] + chw_2, max_dm, np.float32)
         #bz_fdmt_peak_dm, bz_fdmt_peak_time = np.unravel_index(np.argmax(bz_fdmt), bz_fdmt.shape)
         #bz_fdmt_samps_added = bz_fdmt_ones[bz_fdmt_peak_dm, bz_fdmt_peak_time]
         #bz_fdmt_snr = np.max(bz_fdmt) / np.sqrt(bz_fdmt_samps_added)
         
         dx = thefdmt(x)
         fdmt_peak_dm, _ = np.unravel_index(np.argmax(dx), dx.shape)
         fdmt_snr = np.max(dx) / thefdmt.get_eff_sigma(fdmt_peak_dm, 1)

         f.write(f"\t{fdmt_snr / max_snr}")
    print(idm)
    f.write("\n")
f.close()
