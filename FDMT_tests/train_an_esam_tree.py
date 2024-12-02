import sys
sys.path.append("/home/gup037/Codes/ESAM/esam")
import esam, traces

sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests")
import simulate_narrow_frb as snf

from craft import fdmt as FDMT
import numpy as np

def nearest_largest_power_of_2(n):
    power_of_2s = 2**np.arange(0, 30)
    useful_power_of_2s = power_of_2s[power_of_2s > n]
    return useful_power_of_2s[0]



def main():
    nch = 256
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    chw_2 = 0.5
    max_dm = 1000
    nsamps = nearest_largest_power_of_2(max_dm)
    tpulse = nsamps - 20.5
    tx = 1000
    dm_step = 10

    dm_templates = np.arange(0, max_dm, dm_step)

    tree = esam.EsamTree(nch)
    thefdmt = FDMT.Fdmt(f_min= fchans[0], f_off = chw_2 * 2, n_f = nch, max_dt = dm_templates[-1]+1, n_t = nsamps)
            
    product_counts = np.zeros((len(dm_templates), len(tree.count_all_pids())))

    #feed the templates to the tree
    for ii, idm in enumerate(dm_templates):
        x, tot_samps_added = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
        #x = thefdmt.add_frb_track(idm, np.zeros((nch, nsamps)), toffset = 1)
        #trace = traces.digitize_trace(traces.mask_to_trace(x))
        trace = traces.mask_to_trace(x)
        pid = tree.get_trace_pid(trace)
        print(idm)
        product_counts[ii, :] = tree.count_all_pids()
    
    outname = f"esam_tree_with_match_filter_traces_0_{max_dm}_{dm_step}_nch256.pkl"
    #Dumping the filled tree as a pickle
    #np.save("esam_tree_fdmt_tracks_0_4000_1_nch256.pkl", tree)
    np.save(outname, tree)
    np.savetxt("product_counts_for_" + outname + ".txt", product_counts, delimiter=" ")
    print("Dumped to disk as 'product_counts.txt")
main()
