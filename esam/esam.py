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
    def __init__(self, kernel, similarity_score=None):
        self.kernel = kernel
        self.similarity_score = similarity_score
        
    def __str__(self):
        s = f'EndProduct kernel={self.kernel}'
        return s
    
    def __eq__(self, other):
        '''
        Need this otherwise we keep adding the same kernel at the bottom level
        '''    
        iseq = np.array_equal(self.kernel, other.kernel)
        
        if self.similarity_score is not None:
            conv = np.convolve(self.kernel, other.kernel)
            sq_weights = np.sqrt(np.sum(self.kernel**2))
            matched_filter_snr = np.max(conv) / sq_weights
            expected_snr = np.sqrt(np.sum(other.kernel**2))
            if matched_filter_snr / expected_snr > self.similarity_score:
                iseq = True

        return iseq
    
    __repr__ = __str__

    def __call__(self, din, squared_weights = False):
        '''
        Implements the execution of the EndProduct on the data
        din: np.ndarray
            1-D array on which the Endproduct has to be executed
        '''
        out = np.zeros_like(din)
        kernel_size = len(self.kernel)
        for isamp in range(din.size):
            if isamp + kernel_size-1 == din.size:
                break
            if squared_weights:
                weights = self.kernel**2
            else:
                weights = self.kernel
            out[isamp] = np.sum(din[isamp : isamp + kernel_size] * weights)

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
    #print(f"Returning osum = {osum}")
    return osum


class EsamTree:
    def __init__(self, nchan, ichan = 0, similarity_score = 0.9):
        self._products = [] # list containing Products
        self.nchan = nchan
        self._ichan = ichan
        #self.similarity_score = 0.9
        self.similarity_score = None
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
    
    def get_all_pids(self, all_pids = None):
        '''
        Makes a list of lists containing all PIDs for each iteration
        Modifies the provided 'all_pids' list in place
        '''
        if all_pids is None:
            all_pids = [[] for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        if self.nchan == 1:
            all_pids[list_idx].extend(self._products)
        else:
            all_pids[list_idx].extend(self._products)
            self.upper.get_all_pids(all_pids)
            self.lower.get_all_pids(all_pids)

        return all_pids

    def count_all_operations(self, op_counts = None):
        '''
        Counts the number of operations in each iteration and saves them in a list
        '''

        if op_counts is None:
            op_counts = [0 for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        #print(f"{pid_counts}, {type(pid_counts)}, {pid_counts[list_idx]}, {type(pid_counts[list_idx])}")
        if list_idx == 0:
            #Means we are the lowest level - endnodes
            #Then we should not just count the number of products, but how many sums it will do during the convolution phase as well in each product
            for iprod in self._products:
                op_counts[list_idx] += iprod.kernel.size
        else:
            op_counts[list_idx] += len(self._products)

        if self.nchan > 1:
            self.lower.count_all_operations(op_counts)
            self.upper.count_all_operations(op_counts)

        return op_counts


    def count_all_pids(self, pid_counts = None):
        '''
        Counts the number of products in each iteration and saves them in a list
        '''

        if pid_counts is None:
            pid_counts = [0 for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        #print(f"{pid_counts}, {type(pid_counts)}, {pid_counts[list_idx]}, {type(pid_counts[list_idx])}")
        pid_counts[list_idx] += len(self._products)
        
        if self.nchan > 1:
            self.lower.count_all_pids(pid_counts)
            self.upper.count_all_pids(pid_counts)

        return pid_counts


    def get_trace_pid(self, trace) -> int:
        '''
        Returns the product ID for the given trace
        trace:list of nchan values. Each value is tuple(width, offset)'
        '''
        assert len(trace) == self.nchan, f'Unexpedcted trace length in {self}. Was {len(trace)} expected {self.nchan}'
        n2 = self.nchan // 2
        if self.nchan == 1:
            #print(f"Got trace as {trace}, giving trace[0] = {trace[0][1]} to EndProduct")
            prod = EndProduct(trace[0][1], self.similarity_score)
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
        
        #print(f"self.nchan = {self.nchan}, self._ichan = {self._ichan}, trace = {trace}, mid_offset = {mid_offset}, offsets_added_so_far={offsets_added_so_far}") 
        added = False
        if prod not in self._products:
            self._products.append(prod)
            added = True
            
        pid = self._products.index(prod)
        #print(f'{self} of trace {trace}={prod}=PID{pid} added?={added}')
        
        return pid#, cum_offset
        
     
    
    def __call__(self, din, squared_weights = False):
      try:
        assert din.shape[0] == self.nchan
        nt = din.shape[1]
        dout = np.zeros((self.nprod, nt)) # NT here is a bit tricky
        
        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'
            for iprod, prod in enumerate(self._products):
                dout[iprod, :] = prod(din[0], squared_weights)   #din[0] because din is a 1-D data but has 2-D shape (nf, nt) where nf = 1 
        else:
            nf2 = self.nchan // 2 
            lower = self.lower(din[:nf2,...], squared_weights)
            upper = self.upper(din[nf2:,...], squared_weights)
            for iprod, prod in enumerate(self._products):
                #print(f"self.nchan is {self.nchan}, self._ichan is {self._ichan}, prod.offset is {prod.offset}")
                #print(f"upper is {upper}")
                #print(f"lower is {lower}")

                #print(f"Nprod is {self.nprod}")
                
                off = prod.offset
                #if off > 0:
                #    dout[iprod, :off] = upper[prod.pid_upper, :off]
                
                dout[iprod, :] = lower[prod.pid_lower, :]
                if off <= 0:
                    dout[iprod, -off:] += upper[prod.pid_upper, :nt+off]

                    #dout[iprod, -off:] = lower[prod.pid_lower, -off:] + upper[prod.pid_upper, :nt + off]
                elif off > 0:
                    dout[iprod, :nt-off] += upper[prod.pid_upper, off:]
                    
                    
                    #TODO - fix this
                    #raise NotImplementedError
                    #dout[iprod, :nt-off] = upper[prod.pid_upper, :nt-off] + lower[prod.pid_lower, off:] 

                #print(f"dout  is {dout}")
                #print(f"dout.shape is {dout.shape}")
                #plt.figure()
                #plt.imshow(dout, aspect='auto')
                #plt.show()
                #dout[iprod, off:] = lower[prod.pid_lower, 0:nt-off] \
                #        + upper[prod.pid_upper, off:]
        return dout
      except Exception as E:
              import IPython
              IPython.embed()
            
    
def main():
    nchan = 256
    dedisperser = EsamTree(nchan)

    
