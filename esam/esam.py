import numpy as np
import matplotlib.pyplot as plt

class IterProduct:
    def __init__(self, pid_lower, pid_upper, offset):
        self.pid_lower = pid_lower
        self.pid_upper = pid_upper
        self.offset = offset

    def __str__(self):
        s = f"IterProduct with pid_lower={self.pid_lower}, pid_upper={self.pid_upper}, offset={self.offset}"
        return s

    def __repr__(self):
        r = f"l{self.pid_lower}_u{self.pid_upper}_o{self.offset}"
        return r

    def __eq__(self, other):
        if isinstance(other, IterProduct):
            iseq = self.offset == other.offset and\
                    self.pid_lower == other.pid_lower and\
                    self.pid_upper == other.pid_upper
            return iseq

        return False

    def __call__(self, din):
        '''
        Implements the execution of the iteration product on a dataset
        '''
        off = self.offset

class EndProduct:
    def __init__(self, kernel):
        self.kernel = kernel
        
    def __str__(self):
        s = f'EndProduct kernel={self.kernel}'
        return s
    
    def __eq__(self, other):
        '''
        Need this otherwise we keep adding the same kernel at the bottom level
        '''    
        iseq = np.array_equal(self.kernel, other.kernel)
        return iseq
    
    __repr__ = __str__

    def __call__(self, din):
        '''
        Implements the execution of the EndProduct on the data
        din: np.ndarray
            1-D array on which the Endproduct has to be executed
        '''
        #print(type(din), type(self.kernel))
        #print( din.shape, self.kernel.shape)
        out = np.zeros_like(din)
        kernel_size = len(self.kernel)
        for isamp in range(din.size):
            if isamp + kernel_size-1 == din.size:
                break
            out[isamp] = np.sum(din[isamp : isamp + kernel_size] * self.kernel)

        #out =  np.convolve(din, self.kernel, mode='same')
        '''
        plt.figure()
        plt.plot(out)
        plt.show()
        '''
        return out 


def sum_offsets(trace):
    #print(f"sum_offsets got trace {trace}")
    osum = 0
    nchan = len(trace)
    for ichan in range(nchan):
        #print(f"chan in trace is {trace[ichan]}")
        if ichan == 0:
            #If this the 0th channel in the trace, then it's offset wouldn't have been adjusted yet
            pass
        else:
            osum += trace[ichan][0]
    print(f"Returning osum = {osum}")
    return osum


class EsamTree:
    def __init__(self, nchan, ichan = 0):
        self._products = [] # list containing Products
        self.nchan = nchan
        self._ichan = ichan
        assert self.nchan == 1 or self.nchan % 2 == 0, f"EsamTree can only have even number of channels, given = {nchan}"
        
        if nchan == 1:
            self.upper = None
            self.lower = None
        else:
            # normally we'd do FdmtDescriptor here, butif we make a new __class_ if we have a subclass
            # then it'll create the right class
            self.upper = self.__class__(nchan // 2, 2*ichan+1)
            self.lower = self.__class__(nchan // 2, 2*ichan)
        
    def __str__(self):
        s = f'Nchan={self.nchan} chan={self._ichan} nprod={self.ndm}'
        return s
    
    __repr__ = __str__

    @property
    def ndm(self):
        return len(self._products)
    
    @property
    def nprod(self):
        return self.ndm
    
    @property
    def total_products(self):
        '''
        Returns total number of products in hierarchy
        '''
        if self.nchan == 1:
            return self.nprod
        else:
            return self.upper.total_products + self.lower.total_products + self.nprod
        
    def descriptor_tree(self, tree=None, level=0):
        '''
        Returns a list. Each element contains another list. That list contains all descriptors for that iteration
        '''
        if tree is None:
            tree = []
        if self._ichan == 0:
            tree.append([])
            
        tree[level].append(self)
        if self.nchan != 1:
            self.lower.descriptor_tree(tree, level+1)
            self.upper.descriptor_tree(tree, level+1)
        
        return tree              
    
    def get_all_pids(self, all_pids):
        '''
        Makes a list of lists containing all PIDs for each iteration
        Modifies the provided 'all_pids' list in place
        '''

        list_idx = int(np.log2(self.nchan))
        if self.nchan == 1:
            all_pids[list_idx].extend(self._products)
        else:
            all_pids[list_idx].extend(self._products)
            self.upper.get_all_pids(all_pids)
            self.lower.get_all_pids(all_pids)

    
    def get_trace_pid(self, trace) -> int:
        '''
        Returns the product ID for the given trace
        trace:list of nchan values. Each value is tuple(width, offset)'
        '''
        assert len(trace) == self.nchan, f'Unexpedcted trace length in {self}. Was {len(trace)} expected {self.nchan}'
        n2 = self.nchan // 2
        if self.nchan == 1:
            #print(f"Got trace as {trace}, giving trace[0] = {trace[0][1]} to EndProduct")
            prod = EndProduct(trace[0][1])
            mid_offset = None
            offsets_added_so_far = None
            #cum_offset = trace[0][0]
        else:
            pid_lower = self.lower.get_trace_pid(trace[:n2])
            pid_upper = self.upper.get_trace_pid(trace[n2:])
            assert n2 >= 0
            #pid_lower, offset_lower = self.lower.get_trace_pid(trace[:n2])
            #pid_upper, offset_upper = self.upper.get_trace_pid(trace[n2:])
            #assert n2 >= 0
            #cum_offset = offset_lower + offset_upper
            #'''
            mid_offset, _ = trace[n2] # offset and width of the lower of the 2 middle channels
            offsets_added_so_far = sum_offsets(trace[:n2])

            offset = mid_offset + offsets_added_so_far

            assert type(mid_offset) == int, f'offset has wrong type {type(mid_offset)} {mid_offset}'
            #if hasattr(self.lower._products[pid_lower], 'offset'):
            #    offset = mid_offset + self.lower._products[pid_lower].offset
            #else:
            #    offset = mid_offset + 0 
            #'''
            #prod = IterProduct(pid_upper, pid_lower, offset_lower)
            prod = IterProduct(pid_lower, pid_upper, offset)
        
        print(f"self.nchan = {self.nchan}, self._ichan = {self._ichan}, trace = {trace}, mid_offset = {mid_offset}, offsets_added_so_far={offsets_added_so_far}") 
        added = False
        if prod not in self._products:
            self._products.append(prod)
            added = True
            
        pid = self._products.index(prod)
        #print(f'{self} of trace {trace}={prod}=PID{pid} added?={added}')
        
        return pid#, cum_offset
        
     
    
    def __call__(self, din):
        assert din.shape[0] == self.nchan
        nt = din.shape[1]
        dout = np.zeros((self.nprod, nt)) # NT here is a bit tricky
        
        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'
            for iprod, prod in enumerate(self._products):
                dout[iprod, :] = prod(din[0])   #din[0] because din is a 1-D data but has 2-D shape (nf, nt) where nf = 1 
        else:
            nf2 = self.nchan // 2 
            lower = self.lower(din[:nf2,...])
            upper = self.upper(din[nf2:,...])
            for iprod, prod in enumerate(self._products):
                #--print(f"self.nchan is {self.nchan}, self._ichan is {self._ichan}, prod.offset is {prod.offset}")
                #--print(f"upper is {upper}")
                #--print(f"lower is {lower}")

                #print(f"Nprod is {self.nprod}")
                
                off = prod.offset
                #if off > 0:
                #    dout[iprod, :off] = upper[prod.pid_upper, :off]
                
                if off <= 0:
                    dout[iprod, :] = lower[prod.pid_lower, :]
                    dout[iprod, -off:] += upper[prod.pid_upper, :nt+off]

                    #dout[iprod, -off:] = lower[prod.pid_lower, -off:] + upper[prod.pid_upper, :nt + off]
                elif off > 0:
                    #TODO - fix this
                    raise NotImplementedError
                    dout[iprod, :nt-off] = upper[prod.pid_upper, :nt-off] + lower[prod.pid_lower, off:] 
                #--print(f"dout  is {dout}")
                #--print(f"dout.shape is {dout.shape}")
                #plt.figure()
                #plt.imshow(dout, aspect='auto')
                #plt.show()
                #dout[iprod, off:] = lower[prod.pid_lower, 0:nt-off] \
                #        + upper[prod.pid_upper, off:]
            
        return dout                        
    
    
def main():
    nchan = 256
    dedisperser = EsamTree(nchan)

    
