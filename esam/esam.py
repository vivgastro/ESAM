import numpy as np

class IterProduct:
    def __init__(self, pid_lower, pid_upper, offset):
        self.pid_lower = pid_lower
        self.pid_upper = pid_upper
        self.offset = offset

    def __str__(self):
        s = f"IterProduct with pid_lower={self.pid_lower}, pid_upper={self.pid_upper}, offset={self.offset}"
        return s

    def __repr__(self):
        r = f"l{self.pid_lower}_u{self.pid_upper}_o{offset}"

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
        return np.convolve(din, self.kernel, mode='same')

    

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
        s = f'Nchan={self.nchan} chan={self._ichan} ndm={self.ndm}'
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
        if self.ichan == 0:
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
            prod = EndProduct(trace[0])
        else:
            pid_lower = self.lower.get_trace_pid(trace[:n2])
            pid_upper = self.upper.get_trace_pid(trace[n2:])
            assert n2 - 1 >= 0
            mid_offset, _ = trace[n2-1] # offset and width of the lower of the 2 middle channels
            assert type(mid_offset) == int, f'offset has wrong type {type(mid_offset)} {mid_offset}'
            prod = IterProduct(pid_upper, pid_lower, mid_offset)
        
        added = False
        if prod not in self._products:
            self._products.append(prod)
            added = True
            
        pid = self._products.index(prod)
        #print(f'{self} of trace {trace}={prod}=PID{pid} added?={added}')
        
        return pid
        
     
    
    def __call__(self, din):
        assert din.shape[0] == self.nchan
        nt = din.shape[1]
        dout = np.zeros((self.nprod, nt)) # NT here is a bit tricky
        
        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'
            for iprod, prod in enumerate(self._products):
                dout[iprod, :] = prod(din) 
        else:
            nf2 = self.nchan // 2 
            lower = self.lower(din[:nf2,...])
            upper = self.upper(din[nf2:,...])
            for iprod, prod in enumerate(self._products):
                off = prod.offset
                #if off > 0:
                #    dout[iprod, :off] = upper[prod.pid_upper, :off]
                
                dout[iprod, off:] = lower[prod.pid_lower, 0:nt-off] \
                        + upper[prod.pid_upper, off:]
            
        return dout                        
    
    
def main():
    nchan = 256
    dedisperser = EsamTree(nchan)

    
