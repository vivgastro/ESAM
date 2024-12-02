import numpy as np
#from Visibility_injector.inject_in_fake_data import FakeVisibility as FV
import sys
sys.path.append("/home/gup037/Codes/Furby_p3/bin/")
from simulate_injected_blocks import yield_injected_blocks, load_telescope_params_from_file, load_injection_params_from_file

from craft import uvfits, craco_plan, fdmt as FDMT
import craco.search_pipeline
from traces import *
from esam import EsamTree
import argparse
import matplotlib.pyplot as plt

def populate_ESAM_tree_with_masks(esam_tree, masks):
    print(f"populatate_ESAM_tree_with_masks called")# with args esam_tree = {esam_tree}, masks = {masks}"
    trace_ids = []
    pid_counts = []
    for imask, mask in enumerate(masks):
        #print(f"Mask in masks gives {mask}")
        if mask.ndim == 3:
            mask = mask.sum(axis=0).real    #Data must be nbl, nf, nt shape, so sum along the bl axis
            #mask = mask[::-1]
        #plt.figure()
        #plt.imshow(mask, aspect='auto', origin='lower')
        #plt.title(str(imask))
        #plt.show()
        trace = mask_to_trace(mask)
        #print(f"Trace is {trace}")
        trace = digitize_trace(trace)
        #print(f"Digitized trace is {trace}")
        print(f"Calling esam_tree.get_trace_pid for imask {imask}")
        trace_id = esam_tree.get_trace_pid(trace)
        #print(f"Trace id is {trace_id}")
        trace_ids.append(trace_id)
        pid_counts.append(esam_tree.count_all_pids())
        #print(f"PID counts are - ", pid_counts)

    print(f"I should have populated the tree with {len(trace_ids)} trace_ids")
    return trace_ids, pid_counts

def get_fake_block(nf =8 , nt = 16):
    block = np.zeros((nf, nt))
    print(f"Block's shape is - {block.shape}")
    '''
    block[0:2, 10] = 1
    block[2, 9:10] = 1
    block[3:5, 9] = 1
    block[5, 8:9] = 1
    block[5:, 8] = 1
    '''

    block[0, 2] = 1
    block[1, 3:5] = 1
    block[2, 5] = 1
    block[3, 6] = 1
    block[4, 7] = 1
    block[5, 8] = 1
    block[6, 9] = 1
    block[7, 10] = 1



    plt.figure()
    plt.imshow(block, aspect='auto', origin='lower')
    plt.show()

    yield block


def populate_FDMT_ESAM_tree_with_dm_traces(fdmt_esam_tree, thefdmt):
    print(f"populatate_FDMT_ESAM_tree_with_dm_traces called")# with args esam_tree = {esam_tree}, masks = {masks}"
    trace_ids = []
    pid_counts = []
    ndm = thefdmt.max_dt 
    for idm, dm in enumerate(range(ndm)):
        #mask = thefdmt.add_frb_track(dm)
        mask = get_fdmt_track(idm, thefdmt.f_min - thefdmt.d_f/2, thefdmt.d_f, thefdmt.n_f)
        #if idm == 999:
        #    plt.figure()
        #    plt.imshow(mask, aspect='auto', interpolation='None')
        #    plt.show()
        trace = mask_to_trace(mask)
        #print(f"Trace is {trace}")
        trace = digitize_trace(trace)
        #print(f"Digitized trace is {trace}")
        print(f"Calling fdmt_esam_tree.get_trace_pid for idm, dm -  {idm, dm}")
        trace_id = fdmt_esam_tree.get_trace_pid(trace)
        #print(f"Trace id is {trace_id}")
        trace_ids.append(trace_id)
        pid_counts.append(fdmt_esam_tree.count_all_pids())
        #print(f"PID counts are - ", pid_counts)

    print(f"I should have populated the tree with {len(trace_ids)} trace_ids")
    return trace_ids, pid_counts

def main():
    args = get_parser()
    tel_params = load_telescope_params_from_file(args.tel_params_file)
    chw = (tel_params['ftop'] - tel_params['fbottom']) / tel_params['nch']
    fmin = tel_params['fbottom'] + chw/2
    fmax = tel_params['ftop'] - chw/2
    '''
    f = uvfits.open(args.fits_file)
    values = craco.search_pipeline.get_parser().parse_args([])
    values.uv = args.fits_file
    values.ndm = args.ndm

    plan = craco_plan.PipelinePlan(f, values)
    blocker = FV(plan, args.injection_file).get_fake_data_block()
    my_nf = plan.nf
    '''
    my_nf = 256
    my_nt = 256
    blocker = yield_injected_blocks(my_nt, args.injection_file, args.tel_params_file, 999)
    #blocker = get_fake_block(nf = my_nf)

    esam_tree = EsamTree(my_nf)
    fdmt_esam_tree = EsamTree(my_nf)

    trace_ids, pid_counts = populate_ESAM_tree_with_masks(esam_tree, blocker)
    #all_pids = [[] for i in range(int(np.log2(my_nf)) + 1)]
    pid_counts = np.array(pid_counts)
    plt.figure()
    for ii in range(pid_counts.shape[1]):
        plt.plot(pid_counts[:, -1], pid_counts[:, ii], '-.', label=f"Iter{ii}")

    plt.xlabel("No. of traces added in the ESAM tree")
    plt.ylabel("No. of products needed")
    plt.title("Growth of Nproducts with traces")
    #plt.legend()
    #plt.show()
    
    all_pids = esam_tree.get_all_pids()
    pid_counts = esam_tree.count_all_pids()

    #print(f"All pids are:\n{all_pids}")
    print(f"Pid counts are:\n{pid_counts}")
    #print(f"Tree descriptor is {esam_tree.descriptor_tree()}")

    #blocker = FV(plan, args.injection_file).get_fake_data_block()
    blocker = yield_injected_blocks(my_nt, args.injection_file, args.tel_params_file, 999)
    #blocker = get_fake_block(nf = my_nf)
    thefdmt = FDMT.Fdmt(f_min = fmin, f_off = chw, n_f = tel_params['nch'], max_dt =args.ndm, n_t = my_nt, history_dtype=np.float64)
    #thefdmt = FDMT.Fdmt(f_min = plan.fmin, f_off = plan.foff, n_f = plan.nf, max_dt = plan.nd, n_t = plan.nt, history_dtype=np.float64)
    fdmt_trace_ids, fdmt_pid_counts = populate_FDMT_ESAM_tree_with_dm_traces(fdmt_esam_tree, thefdmt)
    
    fdmt_pid_counts = np.array(fdmt_pid_counts)

    plt.gca().set_prop_cycle(None)
    for ii in range(fdmt_pid_counts.shape[1]):
        plt.plot(fdmt_pid_counts[:, -1], fdmt_pid_counts[:, ii], '-', label=f"FDMT-Iter{ii}")
   

    plt.gca().set_prop_cycle(None)
    niters = len(pid_counts)
    ndm = pid_counts[0]
    for jj in range(9):
        plt.plot(np.arange(1, ndm) * 2**(jj+1) + 2**(jj+1), ':', label=f"Brute force iter {jj}")

    plt.legend()
    plt.show()
    
    true_signals = []
    true_snrs = []
    fdmt_signals = []
    fdmt_snrs = []
    esam_signals = []
    esam_snrs = []

    ones = np.ones((my_nf, my_nt))
    ones_fdmt = thefdmt(ones)
    ones_esam = esam_tree(ones)

    for iblock, block in enumerate(blocker):
        #if iblock % 100!= 0:
        #    continue
        print(f"===========================================================================iblock = {iblock}")
        if block.ndim == 3:
            data = block.real.sum(axis=0)
        else:
            data = block
        fdmtout = thefdmt(data)
        esamout = esam_tree(data)


        fdmt_peak_dm, fdmt_peak_time = np.unravel_index(np.argmax(fdmtout), fdmtout.shape)
        esam_peak_dm, esam_peak_time = np.unravel_index(np.argmax(esamout), fdmtout.shape)
        esam_peak_loc = np.argmax(esamout)

        true_signals.append(data.sum())
        true_snrs.append(np.sqrt(np.sum(data**2)))
        fdmt_signals.append(fdmtout.max())
        fdmt_snrs.append(fdmtout.max() / np.sqrt(ones_fdmt[fdmt_peak_dm, fdmt_peak_time]))
        esam_signals.append(esamout.max())
        esam_snrs.append(esamout.max() / np.sqrt(ones_esam[esam_peak_dm, esam_peak_time]))

    
        if args.plot:
            plt.figure()
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(data, aspect='auto')
            ax[0].set_xlabel("time")
            ax[0].set_ylabel("chan")
            ax[0].set_title(f"frb, true_sum = {data.sum()}")
            #'''
            ax[1].imshow(fdmtout, aspect='auto', interpolation='None')
            ax[1].set_xlabel("time")
            ax[1].set_ylabel("fdmt_dm_trial")
            ax[1].set_title(f"fdmt, fdmt_peak = {fdmtout.max()}")
            #'''
            print(esamout.max())
            ax[2].imshow(esamout, aspect='auto', interpolation='None')
            ax[2].set_xlabel("time")
            ax[2].set_ylabel("product id")
            ax[2].set_title(f"esam, esam_peak = {esamout.max()}")
            plt.show()

    true_signals = np.array(true_signals)
    fdmt_signals = np.array(fdmt_signals)
    esam_signals = np.array(esam_signals)
    true_snrs = np.array(true_snrs)
    fdmt_snrs = np.array(fdmt_snrs)
    esam_snrs = np.array(esam_snrs)
    plt.figure()
    plt.plot(true_signals/true_signals, label="True")
    plt.plot(fdmt_signals/true_signals, label="FDMT")
    plt.plot(esam_signals/true_signals, label="ESAM", marker='X')
    plt.legend()
    plt.ylabel("Signal recovery fraction")
    plt.xlabel("DM / trace trial")
    
    
    plt.figure()
    plt.plot(true_snrs/true_snrs, label="True")
    plt.plot(fdmt_snrs/true_snrs, label="FDMT")
    plt.plot(esam_snrs/true_snrs, label="ESAM", marker='X')
    plt.legend()
    plt.ylabel("S/N recovery fraction")
    plt.xlabel("DM / trace trial")
    
    plt.show()



        

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-injection_file", type=str, help="Path to the injection file")
    a.add_argument("-tel_params_file", type=str, help="Path to the injection file", default="/home/gup037/Codes/Furby_p3/example_params/CRACO_params.yml")
    a.add_argument("-fits_file", type=str, help="Path to the fits file that you need to instantiate the plan", default="/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits")
    a.add_argument("-ndm", type=int, help="Ndm for the FDMT to compute", default=256)
    a.add_argument("-plot", action='store_true', help="Show each block", default=False)
    args = a.parse_args()

    return args

if __name__ == '__main__':
    main()


