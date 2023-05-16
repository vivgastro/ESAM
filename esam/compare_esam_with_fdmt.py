import numpy as np
from Visibility_injector.inject_in_fake_data import FakeVisibility as FV
from craft import uvfits, craco_plan, fdmt as FDMT
import craco.search_pipeline
from traces import *
from esam import EsamTree
import argparse

def populate_ESAM_tree_with_masks(esam_tree, masks):
    trace_ids = []
    for mask in masks:
        if mask.ndim == 3:
            mask = mask.sum(axis=0).real    #Data must be nbl, nf, nt shape, so sum along the bl axis
        trace = mask_to_trace(mask)
        trace = digitize_trace(trace)
        trace_id = esam_tree.get_trace_pid(trace)
        trace_ids.append(trace_id)

    return trace_ids



def main():
    args = get_parser()
    f = uvfits.open(args.fits_file)
    values = craco.search_pipeline.get_parser().parse_args([])
    values.uv = args.fits_file
    values.ndm = args.ndm

    plan = craco_plan.PipelinePlan(f, values)
    blocker = FV(plan, args.injection_file).get_fake_data_block()
    esam_tree = EsamTree(plan.nf)
    trace_ids = populate_ESAM_tree_with_masks(esam_tree, blocker)

    blocker = FV(plan, args.injection_file).get_fake_data_block()
    thefdmt = FDMT.Fdmt(f_min = plan.fmin, f_off = plan.foff, n_f = plan.nf, max_dt = plan.nd, n_t = plan.nt, history_dtype=np.float64)
    true_signals = []
    fdmt_signals = []
    esam_signals = []
    for iblock, block in enumerate(blocker):
        data = block.real.sum(axis=0)
        fdmtout = thefdmt(data)
        esamout = esam_tree(data)

        true_signals.append(data.sum())
        fdmt_signals.append(fdmtout.max())
        esam_signals.append(esamout.max())

    
        if args.plot:
            plt.figure()
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(data, aspect='auto')
            ax[0].set_xlabel("time")
            ax[0].set_ylabel("chan")
            ax[0].set_title("frb")

            ax[1].imshow(fdmtout, aspect='auto', interpolation='None')
            ax[1].set_xlabel("time")
            ax[1].set_ylabel("fdmt_dm_trial")
            ax[1].set_title("fdmt")

            ax[2].imshow(esamout, aspect='auto', interpolation='None')
            ax[2].set_xlabel("time")
            ax[2].set_ylabel("product id")
            ax[2].set_title("esam")
            plt.show()

    plt.figure()
    plt.plot(true_signals, label="True")
    plt.plot(fdmt_signals, label="FDMT")
    plt.plot(esam_signals, label="ESAM")



        

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-injection_file", type=str, help="Path to the injection file")
    a.add_argument("-fits_file", type=str, help="Path to the fits file that you need to instantiate the plan", default="/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits")
    a.add_argument("-ndm", type=int, help="Ndm for the FDMT to compute", default=256)
    args = a.parse_args()

    return args

if __name__ == '__main__':
    main()


