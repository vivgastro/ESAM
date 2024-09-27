import sys
sys.path.append("/home/gup037/Codes/ESAM/esam")
import esam, traces

sys.path.append("/home/gup037/Codes/ESAM/FDMT_tests")
import simulate_narrow_frb as snf

from craft import fdmt as FDMT
import numpy as np
import IPython
import argparse


def nearest_largest_power_of_2(n):
    power_of_2s = 2**np.arange(0, 30)
    useful_power_of_2s = power_of_2s[power_of_2s > n]
    return useful_power_of_2s[0]

def evaluate_tree_performance(tree, template):
    '''
    Runs the template through the tree and returns the S/N    
    '''
    try:
        nch, nsamps = template.shape
        dmout = tree(template)
        ones_dmout = tree(np.ones((nch, nsamps)))
        #IPython.embed()

        peak_dm_loc, peak_time_loc = np.unravel_index(np.argmax(dmout), dmout.shape)
        peak_val = dmout[peak_dm_loc, peak_time_loc]
        peak_snr = peak_val / np.sqrt(ones_dmout[peak_dm_loc, peak_time_loc])
        #print(f"Peak val = {peak_val}, Total samps added in ESAM = {ones_dmout[peak_dm_loc, peak_time_loc]}")
    except Exception as e:
        import IPython
        IPython.embed()
    return peak_snr

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-maxdm", type=int, help="MaxDM (def:1000)", default=1000)
    a.add_argument("-dmstep", type=float, help="DM step (def:0,1)", default=0.1)
    a.add_argument("-threshold", type=float, help="Optimisation S/N threshold (def:0.9)", default=0.9)

    args = a.parse_args()
    return args

def main():
    args = get_parser()
    nch = 256
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    chw_2 = 0.5
    max_dm = args.maxdm
    nsamps = nearest_largest_power_of_2(max_dm)
    tpulse = nsamps - 5.5
    tx = 100
    dm_step = args.dmstep
    threshold = args.threshold

    dm_templates = np.arange(0, max_dm, dm_step)

    tree = esam.EsamTree(nch)
    thefdmt = FDMT.Fdmt(f_min= fchans[0], f_off = chw_2 * 2, n_f = nch, max_dt = dm_templates[-1]+1, n_t = nsamps)
            
    #product_counts = np.zeros((len(dm_templates), len(tree.count_all_pids())))

    outbasename = f"final_optimised_esam_tree_fast_with_traces_0_{max_dm}_{dm_step}_nch256_threshold_{threshold}.pkl"

    product_id_to_dm_map = open(f"prod_to_dm_map_for_{outbasename}.txt", 'w')
    product_id_to_dm_map.write(f"prod_id\tDM\n")
    
    product_counts = open(f"product_counts_for_{outbasename}.txt", 'w')
    product_count_header = "N_prods_added\tMax_dm_added\tIter_" + "\tIter_".join(map(str, list(np.arange(int(np.log2(nch)) + 1)))) + "\n"
    product_counts.write(product_count_header)

    brute_force_operation_counts = open(f"operation_counts_for_brute_force_implementation_of_{outbasename}.txt", 'w')
    brute_force_operation_counts_header = "N_templates_added\tMax_dm_added\tN_operations\n"
    brute_force_operation_counts.write(brute_force_operation_counts_header)
    brute_force_operations_counter = 0

    #feed the templates to the tree
    for ii, idm in enumerate(dm_templates):
        nsamps = max(int(idm + 20), 100)
        tpulse = nsamps - 5.5
        x, tot_samps_added = snf.make_pure_frb(nsamps, nch, tx, idm, fchans, chw_2, tpulse)
        #x = thefdmt.add_frb_track(idm, np.zeros((nch, nsamps)), toffset = 1)
        #print(f"Sum = {np.sum(x)}, Total samps added in pure frb = {tot_samps_added}")
        max_snr = np.sum(x) / np.sqrt(tot_samps_added)
        if tree.nprod > 0:
            peak_snr = evaluate_tree_performance(tree, x)
            #print(f"({peak_snr}, {max_snr})")
        else:
            peak_snr = -np.inf
        print(f"{ii} - dm = {idm:.2f} - {(peak_snr / max_snr):.2f}")
        if peak_snr / max_snr < threshold:
            #print(f"ESAM recovers snr less than the threshold --- adding the template")
            trace = traces.mask_to_trace(traces.digitize_as_binary(x))
            pid = tree.get_trace_pid(trace)

            product_id_to_dm_map.write(f"{pid}\t{idm}\n")
            p_counts_str = "\t".join(map(str, tree.count_all_operations()))
            #p_counts_str = "\t".join(format(x, "04g") for x in tree.count_all_operations())
            product_counts.write(f"{pid + 1}\t{idm}\t{p_counts_str}\n")
            #product_counts[ii, :] = tree.count_all_operations()

            brute_force_operations_counter += tot_samps_added - 1
            brute_force_operation_counts.write(f"{pid + 1}\t{idm}\t{brute_force_operations_counter}\n")

        else:
            print(f"Skipped")
            #print(f"Template id {ii} with dm {idm} can be recovered by the tree with {(peak_snr / max_snr) * 100}%  recovery, which is above the threshold - {threshold * 100}%")
    

        #Dump the tree every 100 templates
        if (ii > 0) and (ii % 100 == 0):
            print("Flishing stuff to disk")
            np.save(outbasename, tree)
            product_counts.flush()
            product_id_to_dm_map.flush()

        #trace = traces.digitize_trace(traces.mask_to_trace(x))
        
    #Dumping the filled tree as a pickle
    #np.save("esam_tree_fdmt_tracks_0_4000_1_nch256.pkl", tree)
    np.save(outbasename, tree)
    #np.savetxt("product_counts_for_" + outname + ".txt", product_counts, delimiter=" ")
    print("Dumped to disk")
main()
