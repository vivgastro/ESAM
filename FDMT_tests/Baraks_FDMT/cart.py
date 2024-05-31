
Licence = """
<OWNER> = Barak Zackay (Weizmann Institute of Science)
<YEAR> = 2014

In the original BSD license, both occurrences of the phrase "COPYRIGHT HOLDERS AND CONTRIBUTORS" in the disclaimer read "REGENTS AND CONTRIBUTORS".

Here is the license template:

Copyright (c) 2014, Barak Zackay (Weizmann Institute of Science)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from math import *
from numpy import *
import scipy
import scipy.stats
import scipy.signal
import scipy.io
import time
import os
import operator as op
import glob
import copy
import sys
import inspect
import zipfile
import operator
import itertools

import astropy.io.fits as pyfits
import pylab as P
import matplotlib.cm as cm

MAC = False
LINUX = False
WINDOWS = False
HOME = False
OFFICE = False

TRUE = True
FALSE = False
true = True
false = False
#Desktop_linux = '/home/barakz/Desktop/'
Desktop_windows = r'C:\Users/physics/Desktop/'
DROPBOX = r'C:\Users\physics\Dropbox'

#Dsp = '/home/barakz/DSP/'
#Machine = '/home/barakz/machine_learning'
Dropbox = r'C:\Users\physics\Dropbox'
#

if MAC == True :
    Dropbox = '/Users/barakzackay/Dropbox/'
    execfile(Dropbox +r'/python/utilities/recs.py')

sys.path.append('/Users/barakzackay/Dropbox/python/utilities/')


def pdb(st,debug):
    if debug:
        #print st


##############################################
##############################################
###### PLOTTING TOOLS ########################
##############################################
##############################################

def plot(*args,**kwargs):
    fig1 = P.figure()
    
    p1 = P.plot(*args)
    set_kwargs(kwargs)
    if kwargs.get("show",False) :
        P.show(block = kwargs.get("block",True))
    return PlotCloser(P)

def set_kwargs(kwargs):
    if "xlabel" in kwargs.keys():
        if "fontsize" in kwargs.keys():
            P.xlabel(kwargs["xlabel"],fontsize=kwargs["fontsize"])
        else:
            P.xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs.keys():
        if "fontsize" in kwargs.keys():
            P.ylabel(kwargs["ylabel"],fontsize=kwargs["fontsize"])
        else:
            P.ylabel(kwargs["ylabel"])
    if "xlim" in kwargs.keys():
        P.xlim(kwargs["xlim"])
    if "ylim" in kwargs.keys():
        P.ylim(kwargs["ylim"])
    if "title" in kwargs.keys():
        P.title(kwargs["title"])
    if 'save_loc' in kwargs.keys():
        save_loc = kwargs['save_loc']
        P.savefig(save_loc)

def parity(x):
    if x == 0 :
        return 0
    if x<0 :
        raise Exception("negative number")
    c = 0
    for i in range(int(log2(x))+2):
        c+= (x&1)
        x = (x>>1)
    return c%2

# yields all ordered different pairs of a list L
# for non-oredered pairs use cartesian



def plot_many(*args,**kwargs):
    if len(args)==2:
        x = args[0]
        ys = args[1]
    if len(args)==1:
        ys = args[0]
        x = range(len(ys[0]))
        assert len(list(set(lens(ys))))==1
    if len(args)>2 :
        raise AssertionError("only 2 non keyword arguments required")
    fig1 = P.figure()
    plots = []
    for y in ys :
        plots += P.plot(x,y)
    
    labels = kwargs.get('labels',['plt_%d'%i for i in range(len(ys))])
    P.legend(plots,labels)
    set_kwargs(kwargs)
    
    if LINUX :
        P.show()
    return PlotCloser(P)

def plot_together(plot_args,d1=False,d2=False,plot_function = "plot",title = None,plot_kwargs = {},**kwargs):

    nplots = len(plot_args)
    if False in [d1,d2]:
        d1 = len(plot_args)
        d2 = 1
    fig = P.figure()
    if title!=None :
        fig.suptitle(title,fontsize = 28)
    for i in range(d1):
        for j in range(d2):
            #print i,j
            if (i+d1*j)>= nplots :
                continue
            ax = fig.add_subplot(d1,d2,1 + i+d1*j)
            if plot_function == "plot":
                l = ax.plot(*plot_args[i+d1*j])
            if plot_function == "cview":
                cax = ax.imshow(plot_args[i+d1*j],interpolation = 'nearest')
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
            if plot_function == "plot_histogram":
                plot_histogram(plot_args[i+d1*j][0],plot_args[i+d1*j][1],P_obj = ax,**plot_kwargs)

    set_kwargs(kwargs)
    if LINUX:
        P.show()
    return PlotCloser(P)


def cview(nums,cmap = "gray", show = False, block = False, flip = True, colorbar = True, extent = False,xticklabels = [], yticklabels = [],**kwargs):
    
    fig = P.figure()
    ax = fig.add_subplot(111)
    if cmap == "gray":
        if not extent:
            cax = ax.imshow(nums, interpolation='nearest',cmap=cm.gray)
        else:
            cax = ax.imshow(nums, interpolation='nearest',cmap=cm.gray, extent = extent)
    if cmap == "color":
        if not extent:
            cax = ax.imshow(nums, interpolation='nearest')
        else:
            cax = ax.imshow(nums, interpolation='nearest', extent = extent)
    
    if xticklabels != []:           
        ax.set_xticklabels(xticklabels)
    if yticklabels != []:
        ax.set_yticklabels(yticklabels)
    if colorbar:
        cbar = fig.colorbar(cax)
    if flip:
        ##print "flipping!"
        fig.gca().invert_yaxis()
    set_kwargs(kwargs)
    if show:
        if block :
            P.show(block = True)
        else:
            P.show(block = False)
    return PlotCloser(P)

def star_view(nums,log_scale = True,vmin = 8,vmax = 64,**kwargs):
    fig = P.figure()
    ax = fig.add_subplot(111)
    if log_scale :
        cax = ax.imshow(log2(abs(nums)+0.01),interpolation = 'nearest',vmin = log2(vmin),vmax = log2(vmax), cmap = cm.gray)
    else :
        cax = ax.imshow(abs(nums),interpolation = 'nearest',vmin = vmin,vmax = vmax, cmap = cm.gray)
        
    set_kwargs(kwargs)
    P.show(block = True)
    return PlotCloser(P)

def scatter_plot3d(x,y,z):
    from mpl_toolkits.mplot3d import Axes3D
    #import matplotlib.pyplot as plt
    fig = P.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    P.show()
    return PlotCloser(P)
    



#################################################################
#################################################################
##### usefull functions #########################################
#################################################################
#################################################################

def flen(x):
    return float(len(x))
    
def out(x,num = True):
    for i,y in enumerate(x) :
        if num == True:
            #print i,':\t',y
        else:
            #print y

def makbil(x,tabs = 1,groups = 'default'):
    if groups == 'default':
        groups = len(x[0])
    for j in range(max(1,len(x[0])/groups)):
        for i,y in enumerate(x):
            #print i,':\t',
            for k,t in enumerate(y[j*groups:(j+1)*groups]):
                #print t,
                if (k%tabs) == tabs-1:
                    #print '\t',
            #print 
        
    
        

def #print_table(T , headlines,max_width = 5):
    #print '#\t',
    for i in headlines:
        #print i,'\t\t',
    #print 
    #print '----------------------------------------------------------------'
    for ind,x in enumerate(T) :
        #print ind,'\t',
        for i in x :
            if type(i) == list and (len(list(set(map(type,i)))) == 1) and type(i[0])==float64 :
                #print st_floats(i),'\t\t',
            else :
                #print str(i)[:max_width],'\t\t',
        #print


def local_maximums(x):
    return [i for i in range(1,len(x)-1) if x[i]>=x[i-1] and x[i]>=x[i+1]]

def local_maximums_2d(ar):
    """
    returns all the locations that are greater than their immediate coordinates (including diagonals)
    """
    return array(zip(*nonzero((ar[1:-1,1:-1]>ar[2:,1:-1])&(ar[1:-1,1:-1]>ar[0:-2,1:-1])& (ar[1:-1,1:-1]>ar[1:-1,2:])&(ar[1:-1,1:-1]>ar[1:-1,0:-2])&(ar[1:-1,1:-1]>ar[2:,2:])&(ar[1:-1,1:-1]>ar[2:,0:-2])&(ar[1:-1,1:-1]>ar[0:-2,0:-2])&(ar[1:-1,1:-1]>ar[0:-2,2:]))))+1

def significant_local_maximum_2d(ar,gap_units = 10):
    ar_sig = median(abs(ar-mean(ar)))
    ar2 = ar+gap_units*ar_sig
    return array(zip(*nonzero((ar[1:-1,1:-1]>ar2[2:,1:-1])&(ar[1:-1,1:-1]>ar2[0:-2,1:-1])& (ar[1:-1,1:-1]>ar2[1:-1,2:])&(ar[1:-1,1:-1]>ar2[1:-1,0:-2])&(ar[1:-1,1:-1]>ar2[2:,2:])&(ar[1:-1,1:-1]>ar2[2:,0:-2])&(ar[1:-1,1:-1]>ar2[0:-2,0:-2])&(ar[1:-1,1:-1]>ar2[0:-2,2:]))))+1

def significant_local_maximum_2d_map(ar,gap_units = 10):
    ar_sig = median(abs(ar-mean(ar)))
    ar2 = ar+gap_units*ar_sig
    return (ar[1:-1,1:-1]>ar2[2:,1:-1])&(ar[1:-1,1:-1]>ar2[0:-2,1:-1])& (ar[1:-1,1:-1]>ar2[1:-1,2:])&(ar[1:-1,1:-1]>ar2[1:-1,0:-2])&(ar[1:-1,1:-1]>ar2[2:,2:])&(ar[1:-1,1:-1]>ar2[2:,0:-2])&(ar[1:-1,1:-1]>ar2[0:-2,0:-2])&(ar[1:-1,1:-1]>ar2[0:-2,2:])
   
#def DB(x):
#    return 10*log10(x)


def st_floats(x):
    st = '['
    for i in x :
        st += ('%3f'%i)[:6] + ', '
    st = st[:-2]
    st += ']'
    return st

def probify(arr):
    return arr/float(sum(arr))

if "PLOT_CLOSER_ind" not in globals().keys():
    PLOT_CLOSER_ind = []

def plot_histogram(nums,bins=100,probify_ = False,fit_normal = False,P_obj = False,show = False,block = True,**kwargs):
    dont_close = False
    if P_obj == False :
        dont_close = False
        P_obj = P
        fig1 = P_obj.figure()
    hist = histogram(nums,bins)
    centers = (hist[1][1:] + hist[1][:-1])/2.
    width = hist[1][1] - hist[1][0]
    a = hist[0]
    if probify_ :
        a = probify(a)
    
    if fit_normal != False :
        ##print "fitting normal"
        E = median(nums)
        # E = mean(nums) is less stable but more accurate
        sigma = sqrt(var(nums))
        n = scipy.stats.norm.cdf((hist[1]-E)/sigma)
        nn = n[1:]-n[:-1]
        ##print E,sigma
        ##print centers,nn
        P.plot(centers,nn*len(nums))
    p1 = P_obj.bar(hist[1][:-1], a,   width, color='r')
    set_kwargs(kwargs)
    if dont_close :
        return
    if show:
        P.show(block = block)
    return PlotCloser(P)


class PlotCloser(object):
    def __init__(self,P):
        self.P = P
        self.index = random.randint(0,2**20)
        globals()['PLOT_CLOSER_ind'].append(self.index)
        if HOME :
            P.show()
    def __del__(self):
        if globals()['PLOT_CLOSER_ind'][-1] == self.index:
            for i in range(len(globals()['PLOT_CLOSER_ind'])):
                self.P.close()
                globals()['PLOT_CLOSER_ind'] = []

#######################
##### Iterating tools #
#######################

class LenGen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self): 
        return self.length

    def __iter__(self):
        return self.gen

def cartesian(*args):
    return LenGen(itertools.product(*args),reduce(operator.mul,map(len,args)))

def combinations(L,k):
    return LenGen(itertools.combinations(L,k),over(len(l),k))

pairs = lambda L:combinations(L,2)

def progress_bar(l):
    t = time.time()
    n = flen(l)
    seconds = 0
    for i,obj in enumerate(l) :
        yield obj
        t0 = time.time()
        if (t0-t)>seconds :
            seconds = int(t0-t)
            time_left = (t0-t)/float(i+1) * (n-(i+1) )
            if time_left<60 :
                # not leaving any trace of former writing...
                ##print "\t\t\t\t\t\t\t\t\t\r",
                #print "\riteration number %d/%d "%(i,int(n-1))+"\t\ttime left: %d seconds"%time_left ,
            elif time_left<3600 :
                # not leaving any trace of former writing...
                ##print "\t\t\t\t\t\t\t\t\t\r",
                #print "\riteration number %d/%d "%(i,int(n-1))+"\t\ttime left: %d minutes and %d seconds"%(time_left/60,time_left%60) ,
            elif time_left<3600*24:
                # not leaving any trace of former writing...
                ##print "\t\t\t\t\t\t\t\t\t\r",
                #print "\riteration number %d/%d "%(i,int(n-1))+"\t\ttime left: %d hours %d minutes and %d seconds"%(time_left/3600,(time_left%3600)/60,time_left%60),
            elif time_left>3600*24:
                # not leaving any trace of former writing...
                ##print "\t\t\t\t\t\t\t\t\t\r",
                #print "\riteration number %d/%d "%(i,int(n-1))+"\t\ttime left: %d days %d hours %d minutes and %d seconds"%(time_left/(3600*24),(time_left%(3600*24))/3600,(time_left%3600)/60,time_left%60),

tqdm = progress_bar
trange = lambda x:progress_bar(xrange(x))   
lens = lambda x:map(len,x)
lrange = lambda x:xrange(len(x))

def MS(vec):
    return mean(vec**2)

def fact(i):
    return reduce(op.mul,range(1,i+1),1)

def over(n,m):
    return fact(n)/(fact(m)*fact(n-m))

def log2_fact(i):
    return sum([log2(i) for i in range(1,i+1)])

def log2_over(n,m):
    return log2_fact(n)-log2_fact(n)-log2_fact(n-m)

def time_it(command):
    t = time.time()
    exec(command)
    t = time.time()-t
    #print "the command took :",t

def sum_list(x):
    return reduce(lambda z,y:z+y,x)
#def blockify(x,k):
#    return [x[i*k:(i+1)*k] for i in range(len(x)/k)]

def delta_sub(x,dist = 1):
    xx = array(x)
    return xx[dist:]-xx[:-dist]
        
def read_fits_file(file_name,sec_x = None,sec_y = None):
    hdu = pyfits.open(file_name)
    if sec_x == None or sec_y == None:
        return hdu[0].data
    else :
        return hdu[0].section[sec_x[0]:sec_x[1],sec_y[0]:sec_y[1]]

def fits_save(data,file_name):
    hdu = pyfits.PrimaryHDU(data)
    #hdu_list = pyfits.HDUList([hdu])
    hdu.writeto(file_name)
    return

def maxnd(ar):
    return ar[argmaxnd(ar)]
def argmaxnd(ar):
    return unravel_index(argmax(ar),ar.shape)

def gaussian2d(size,x0,y0,sigma):
    x,y = mgrid[:size,:size]
    return (1/(pi*sigma**2)) * e**(-((x-x0)**2 +(y-y0)**2)/float(sigma**2))

def mifkad(l):
    d = {}
    for i in l :
        d[i] = d.get(i,0)+1
    return d

def max_arrays(a,b):
    return (a<b)*b + (a>=b)*a

def blockify(a,size):
    return [a[size*i:size*(i+1)] for i in range(len(a)/size)]

def pgc(func):
    #print inspect.getsource(func)

def extract_zip_file_names(zip_file_name):
    z = zipfile.ZipFile(file(zip_file_name))
    names = z.namelist()
    z.close()
    return names

def extract_zip_files(zip_file_name,names):
    z = zipfile.ZipFile(file(zip_file_name))
    data = [z.read(name) for name in names]
    z.close()
    return data
print("Cart has been successfully loaded")
# magic line that makes the interactive plotting available again
# %pylab qt


