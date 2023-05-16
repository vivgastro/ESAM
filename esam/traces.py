import numpy as np

def digitize_array(arr, levels):
    level_idxs = np.digitize(arr, levels) -1    #np.digitize seemed to return 1-indexed indices of levels #TODO double check this works
    return levels[level_idxs]


def digitize_as_binary(arr):
    return np.ones_like(arr)


def mask_to_trace(mask):
    '''
    Takes a 2-D numpy array containing the cells that have to be summed
    and makes a trace out of it
    '''
    trace = []
    min_start = 0
    for ichan, chan_data in enumerate(mask):
        idxs = np.where(chan_data > 0)[0]
        assert len(idxs) > 0, f"Mask cannot be empty! Found to be empty for ichan {ichan}"
        start, end = idxs[0], idxs[-1]
        kernel = chan_data[idxs[0]:idxs[-1] + 1]

        if ichan ==0:
            min_start = idxs[0]
            offset = 0
        else:
            offset = int(idxs[0] - min_start )

        trace.append([offset, kernel])

    return trace

def digitize_trace(trace, nbits=1):
    '''
    Digitizes the trace based on how many nbits you want
    '''
    levels = np.arange(2**nbits)
    levels = np.insert(levels, 0, 0)    #Prepends 0
    for ichan in range(len(trace)):
        off, kernel = trace[ichan]
        kernel = digitize_as_binary(kernel)
        #kernel = np.sign(kernel)    #Digitizes to -1, 0 and 1 only
        #kernel[kernel < 0] = 0      #Fixes the -1s
        trace[ichan] = [off, kernel]
    
    return trace

def trace_to_mask(trace, padding = 0):
    '''
    Converts a given trace into a 2-D mask (numpy array)
    Optionally pads the mask with 'padding' zeros on both sides of the trace
    '''
    nchans = len(trace)

    pass








