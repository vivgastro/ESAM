import numpy as np

def digitize_array(arr, levels):
    level_idxs = np.digitize(arr, levels) -1    #np.digitize seemed to return 1-indexed indices of levels #TODO double check this works
    return levels[level_idxs]


def digitize_as_binary(arr):
    d = np.zeros_like(arr)
    d[arr > 0] = 1
    return d

'''
def mask_to_trace(mask):
    #'
    #Takes a 2-D numpy array containing the cells that have to be summed
    #and makes a trace out of it
    #'
    #print(f"mask_to_trace got mask =\n{mask}")
    trace = []
    prev_off = 0
    nchan = len(mask)
    for ichan, chan_data in enumerate(mask):
        idxs = np.where(chan_data > 0)[0]
        assert len(idxs) > 0, f"Mask cannot be empty! Found to be empty for ichan {ichan}"
        start, end = idxs[0], idxs[-1]
        kernel = chan_data[idxs[0]:idxs[-1] + 1]
       
        if ichan == 0:
            prev_off = idxs[0]

        offset = int(idxs[0] - prev_off)
        prev_off = idxs[0]

        trace.append([offset, kernel])
    #trace[0][0] = 0
    #print(f"mask_to_trace is returning trace {trace}")
    return trace

'''

def mask_to_trace(mask, convention='start'):
    '''
    Takes a 2-D numpy array containing the cells that have to be summed
    and makes a trace out of it
    '''
    trace = []
    prev_off = 0
    if convention == 'end':
        offset_marker = -1
    elif convention == 'start':
        offset_marker = 0
    else:
        raise Exception(f"Unknown convention provided - convention")
    nchan = len(mask)
    for ichan, chan_data in enumerate(mask):
        idxs = np.where(chan_data > 0)[0]
        assert len(idxs) > 0, f"Mask cannot be empty! Found to be all zeros for ichan {ichan}"
        kernel = chan_data[idxs[0]:idxs[-1] + 1]

        if ichan == 0:
            prev_off = idxs[offset_marker]

        offset = int(idxs[offset_marker] - prev_off)
        prev_off = idxs[offset_marker]

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


def trace_to_mask(trace):
    '''
    Converts a given trace into a 2-D mask (numpy array)
    '''
    nchan = len(trace)
    min_pos = 0
    max_pos = 0
    curr_chan_start = 0
    for ichan in range(nchan):
        curr_chan_start += trace[ichan][0]
        print(ichan, curr_chan_start, trace[ichan][0])
        curr_chan_end = curr_chan_start + len(trace[ichan][1]) - 1
        
        if curr_chan_start < min_pos:
            min_pos = curr_chan_start
        if curr_chan_end > max_pos:
            max_pos = curr_chan_end
    print("min_pos, max_pos are", min_pos, max_pos)
    nt = max_pos - min_pos + 1
    mask = np.zeros((nchan, nt))
    start_pos = np.abs(min_pos)
    for ichan in range(nchan):
        kernel = trace[ichan][1]
        start_pos += trace[ichan][0]   #will be 0 for chan0 by definition
        mask[ichan, start_pos:start_pos + len(kernel)] = kernel
        
    return mask


def cff(f1, f2, fmin, fmax):
    return (f1**-2 - f2**-2) / (fmin**-2 - fmax**-2)


def fround(num, rounding='nearest'):
    if rounding == 'floor':
        return int(np.floor(num))
    if rounding == 'ceil':
        return int(np.ceil(num))
    if rounding == 'nearest':
        return int(np.around(num))


def get_tstart_and_dm_in_samps_for_a_band(tstart, tend):
    '''
    tstart: float
        The time at which the pulse enters the channel from lower freq edge
    tend: float
        The time at which the pulse leaves the channel from higher freq edge
        
    This convention implies that tstart > tend
    
      1     2     3     4
    __|_____|_____|_____|__
    ...\.................
    ....\................
    .....\...............
    ......\..............
    __|_____|_____|_____|__
      1     2     3     4
    
    
    In the above diagram ~'1' is the tend and ~'1.9' is the tstart
    
      1     2     3     4
    __|_____|_____|_____|____
    ......\.................
    .......\................
    ........\...............
    .........\..............
    __|_____|_____|_____|____
      1     2     3     4
    
    In this example, ~'1.5' is tend and ~'2.2' is tstart
    
    '''
    
    if np.floor(tstart) == np.floor(tend):
        tstart_samp = int(np.floor(tstart))
        dm_samp = 0
    else:
        full_samples = np.floor(tstart) - np.ceil(tend)

        fractional_start_samp = tstart - np.floor(tstart)
        fractional_end_samp = 1- (tend - np.floor(tend))
        
        critical_fraction = np.sqrt(full_samples * (full_samples + 1)) - full_samples
        print(tstart, tend, full_samples, fractional_start_samp, fractional_end_samp, critical_fraction)
        #critical_fraction = 0            
        if fractional_start_samp >= critical_fraction:
            tstart_samp = int(tstart)
        else:
            tstart_samp = int(tstart) -1
                        
        if fractional_end_samp >= critical_fraction:
            tend_samp = int(tend)
        else:
            tend_samp = int(tend) + 1
                        
        dm_samp = tstart_samp - tend_samp
    
    return tstart_samp, dm_samp


def find_optimal_samples_to_add(pulse_start, pulse_end, optimize = 'signal'):
    '''
    pulse_start: float
        Time in floats at which the pulse enters the channel from lower edge
    pulse_end: float
        Time in floats at which the pulse leaves the channel from upper edge
    optimize: str
        Optimize for snr or signal
    '''

    if np.floor(pulse_start) == np.floor(pulse_end):
        tstart_samp = int(np.floor(pulse_start))
        dm_samp = 0
    else:
        full_samples = np.floor(pulse_start) - np.ceil(pulse_end)
        fractional_start_samp = pulse_start - np.floor(pulse_start)
        fractional_end_samp = np.ceil(pulse_end) - pulse_end

        total_duration = pulse_start - pulse_end
        fractional_start_duration = fractional_start_samp / total_duration
        fractional_end_duration = fractional_end_samp / total_duration
        full_duration = full_samples / total_duration

        
        case_all_snr = np.sqrt(full_duration**2 + fractional_start_duration**2 + fractional_end_duration**2) / np.sqrt(full_samples + 2)
        case_left_snr = np.sqrt(full_duration**2 + fractional_end_duration**2) / np.sqrt(full_samples + 1)
        case_right_snr = np.sqrt(full_duration**2 + fractional_start_duration**2) / np.sqrt(full_samples + 1)
        if full_samples > 0:
            case_neither_snr = np.sqrt(full_duration**2) / np.sqrt(full_samples)
            cases = [case_left_snr, case_all_snr, case_right_snr, case_neither_snr]
        else:
            cases = [case_left_snr, case_all_snr, case_right_snr]

        if optimize == 'snr':
            argmax = np.argmax(cases)
        elif optimize == 'signal':
            argmax = 1
        else:
            raise ValueError("Optimize needs to be snr/signal")

        print(cases, argmax)
        if argmax == 0:
            tstart_samp = int(np.floor(pulse_start-1))
            dm_samp = int(tstart_samp - np.floor(pulse_end))
        elif argmax == 1:
            tstart_samp = int(np.floor(pulse_start))
            dm_samp = int(tstart_samp - np.floor(pulse_end))
        elif argmax == 2:
            tstart_samp = int(np.floor(pulse_start))
            dm_samp = int(tstart_samp - np.ceil(pulse_end))
        elif argmax == 3:
            tstart_samp = int(np.floor(pulse_start-1))
            dm_samp = int(tstart_samp - np.ceil(pulse_end))

    return tstart_samp, dm_samp


def get_fdmt_track(dm_samps, f0, chw, nch, rounding = 'nearest'):
    '''
    dm_samps - Delta T of frb across the whole band in samps (top edge to bottom edge)
    chw - channel bandwidth (always positive)
    f0 - central freq of 0 (1st) channel
    nch - no of channels
    rounding - convention which tells whether the deltaT value (e.g. 3.5) rounds to 3 (floor), 4 (ceil), or 4 (nearest)
    '''
    track = np.zeros((nch, dm_samps + 1))
    fmin = f0 - chw/2
    fmax = fmin + nch * chw
   
    spp = 0.5
    print(fmin, fmax)
    fill_dm_track(dm_samps, fmin, fmax, track, chan_index = 0, nch = nch, delay = dm_samps + spp)
    return track

def fill_dm_track(dm_whole_band, fmin, fmax, track, chan_index, nch, delay, rounding = 'nearest'):

    if nch == 1:
        #tstart_samp, dm_samp = get_tstart_and_dm_in_samps_for_a_band(delay, delay - dm_whole_band)
        tstart_samp, dm_samp = find_optimal_samples_to_add(delay, delay - dm_whole_band)
        print('~', chan_index, dm_whole_band, delay, tstart_samp, dm_samp)

        #track[chan_index, fround(delay - dm_whole_band):fround(delay)+1] += 1
        track[chan_index, tstart_samp - dm_samp: tstart_samp + 1] += 1

    else:
        print('          +', nch, chan_index, dm_whole_band, delay)
        fmid = (fmin + fmax) / 2
        
        dm_upper_band = dm_whole_band * cff(fmid, fmax, fmin, fmax)
        dm_lower_band = dm_whole_band * cff(fmin, fmid, fmin, fmax)
        print('          +', nch, chan_index, dm_whole_band, delay, dm_upper_band, dm_lower_band)

        fill_dm_track(dm_lower_band, fmin, fmid, track, 2*chan_index, nch //2, delay)
        fill_dm_track(dm_upper_band, fmid, fmax, track, 2*chan_index + 1, nch //2, delay - dm_lower_band)


def get_dm_curve(dm_samps, f0, chw, nch, spp = 0.5):
    x = np.arange(0, dm_samps, 0.0001)
    freqs = f0 + np.arange(nch) * chw
    y = np.arange(freqs[-1] + chw/2, freqs[0] - chw/2, -0.0001)

    t = y**-2 
    t -= t[0]
    t /= t.max()
    t *= dm_samps
    t += spp
    return t, (y - y[-1]) / chw


