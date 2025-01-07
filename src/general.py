## These are general .py programs written to be used across all programs ##
# 2020-01-09 JS

import numpy as np
import os

def listUnion(li1, li2): # union of two lists
    return list(set().union(li1,li2))
                
def listDiffs(li1, li2): # get differences between two lists
    return list(list(set(li1)-set(li2)) + list(set(li2)-set(li1)))

def listIntersect(li1, li2): # intersection of two lists
    return [value for value in li1 if value in li2]

def isNaN(num):
    # handles strings, float8, and float64 unlike np.isnan
    initial_test = num != num
    second_test = num == 'nan' # some string types don't work with above line
    return initial_test | second_test

def nameAsString(string):
    for k, v in list(locals().iteritems()):
         if v is string:
            name_as_str = k
    return name_as_str
def splitUpString(string,delimiter=''):
    if delimiter == '':
        split_array = np.array(list(map(float,string.split())))
    else:
        split_array = np.array(list(map(float,string.split(delimiter))))
    return split_array

def atoi(text):
    # for natural_keys below
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    # sort strs in human order
    example: alist.sort(key=natural_keys) sorts in human order
    '''
    import re
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def CMLReadDFRow(row):
    from cmlreaders import CMLReader, get_data_index
    ''' # Use like this:
        for row in df.itertuples():
            reader = CMLReadDFRow(row)
    '''
    rd = row._asdict() # this takes df and takes values from 1 row as a dict
    return CMLReader(rd['subject'], rd['experiment'], rd['session'], \
                     montage=int(rd['montage']), localization=int(rd['localization']))
    # dirty secret: Readers reads: eegoffset, experiment, subject, and eegfile...but really should
    # pass in sessions since sampling rate could theoretically change...
    
def set_pubfig():
    # seaborn parameters for publication figures
    import seaborn as sb
    import matplotlib.pyplot as plt
    sb.set_context("paper", rc={"font.size":14,"axes.titlesize":14,"axes.labelsize":14,
                                "axes.ticksize":12,"axes.titlesize":14})
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.legend(fontsize=14)

def plotHistoBar(values,start,end,bin_size,tick_range_divisor=1,normalize=False,log_scale=False):
    # e.g. plotHistoBar(lengths,0,0.2,0.01,tick_range=np.arange(0,0.2,0.05),normalize=True)
    # properly plot a histogram (with the values shown in the current bins!)
    # input: values: what to create histogram of
    #        start, end, and bin_size: numbers to define an np.arange
    #        normalize: True if you want proportion out of 1
    #        tick_range: np.arange where you wanted labeled ticks
    import matplotlib.pyplot as plt
    bins = np.arange(start,end+bin_size+bin_size/1000,bin_size) # added bin+0.001 to show last bin and last tick
    if log_scale == True:
        bins = np.logspace(np.log10(bins[1]),np.log10(bins[-1]),len(bins))
        
    hist = np.histogram(values,bins)
    if normalize == True:
        yvalues = hist[0]/sum(hist[0])
    else:
        yvalues = hist[0]
    xr = (bins[1:]+bins[:-1])/2
    ax = plt.bar(xr,yvalues,width=0.8*bin_size)

    # get ticks
    wanted_ticks = np.arange(bins[0],bins[-1]+bin_size/1000,bin_size*tick_range_divisor)
    wanted_ticks = np.around(wanted_ticks,3) # for some reason arange loses precision sometimes
    ticks = []
    has_ticks = []
    for tick in bins:
        if tick in wanted_ticks:
            ticks.append(tick)
            has_ticks.append(True)
        else:
            has_ticks.append(False)
    plt.xticks(xr[has_ticks[:-1]]-bin_size/2,ticks) # [:-1 since took midpoints to get xr above]
    if log_scale == True:
        plt.xscale('log')
        
def plotLogHist(x, bins):
    import matplotlib.pyplot as plt
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    
def tscorePlot(mdl,yrange=7.5,names=None): 
    # plot the tscores with 2 SEs shaded for model results
    import matplotlib.pyplot as plt
    if names is None: # need to add names for regression results without column names like MixedLMresults
        try:
            names = np.array(mdl.normalized_cov_params.columns)
        except:
            raise('You need some names for your xticks')
    plt.subplots(1,1, figsize=(4,3))
    xr = np.linspace(-.5,len(mdl.tvalues)-0.5,100)
    twoSD = 1.96*np.ones(100)
    plt.fill_between(xr,twoSD,-twoSD,alpha=0.1,color=[0,0,0])
    xticks = np.arange(0,len(mdl.tvalues)-0.5)
    plt.bar(xticks,mdl.tvalues)
    ax = plt.gca(); ax.set_ylim((-yrange,yrange))
    plt.ylabel('t-values')
    plt.xticks(xticks,names,rotation=90)
    plt.show()
    
def csvWriter(lists,filename):
    # write lists of info into separate lines of comma delmited format for opening in excel
    import csv
    with open(str(filename)+'.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', # comma separate
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in lists:
            csvwriter.writerow(t)

def seFromProp(num_correct,trials):
    # calculate standard error for proportions
    stderr = np.sqrt(num_correct*(1-num_correct/trials)/(trials-1)) / np.sqrt(trials)
    return stderr

def findInd(idx): # note: np.where does this, but it returns a tuple and this returns a list...actually now an array
    # get the indices when given boolean vector (like find function in matlab)
    idxs = [i for i,val in enumerate(idx) if val]
    return np.array(idxs)

def findAinB(A,B): # find where A is in B. this version works for lists and np arrays
    temp = set(A)
    inds = [i for i, val in enumerate(B) if val in temp] 
    return inds

def findAinBlists(A,B):
    # gets indicies for A in B where B is a list of lists
    inds = []
    for first in A:
        temp_ind = findInd(first in sublist for sublist in B)
        inds.extend(temp_ind)
    inds = np.unique(inds)
    return inds

def superVstack(a,b):
    # make it so you can vstack onto empty row
    if len(a)==0:
        stack = b
    elif len(b)==0:
        stack = a
    else:
        stack = np.vstack([a,b])
    return stack

def findUniquePairs(a):
    # take a list of number pairs and return the unique pairs
    a = np.array(a)
    a.sort(axis=1)
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
        )
    _,d,cts = np.unique(b,return_index=True,return_counts=True)
    uniquePairs = a[d] 
    num_of_each = cts
    return uniquePairs,num_of_each

def addOnesColumn(X):
    # add a row of ones at the beginning of a np array for a regression
    if np.ndim(X)==1: # if a single column
        X = np.vstack((np.ones(len(X)),X)).T
    else:
        X = np.hstack((np.ones(len(X))[:,np.newaxis],X))
    return X

def fileDeleter(path,partial_name):
    import os,glob
    for filename in glob.glob(path+partial_name+"*"):        
        os.remove(filename)
'''
use these to delete the files made by ClusterRun:
fileDeleter("/home1/john/thetaBurst/code/","sge_engine")
fileDeleter("/home1/john/thetaBurst/code/","sge_controller")
fileDeleter("/home1/john/thetaBurst/code/","bcbio-e")
'''

def getLogicalChunks(array):
    # find start and end indices for chunks of values >0 in an array
    foundstart = False
    foundend = False
    startindex = 0
    endindex = 0
    starts = []
    ends = []    
    for i in range(0, len(array)):
        if array[i] != 0:
            if not foundstart:
                foundstart = True
                startindex = i
        else:
            if foundstart:
                foundend = True
                endindex = i - 1
        # for last point
        if i==(len(array)-1):
            if foundstart:
                starts.append(startindex)
                if array[i] == 1:
                    ends.append(i)
                else:
                    ends.append(i-1)
        if foundend:
            #print(startindex, endindex)
            starts.append(startindex)
            ends.append(endindex)
            foundstart = False
            foundend = False
            startindex = 0
            endindex = 0  
    return starts,ends

def bootstrap(data, bootnum=100, samples=None, bootfunc=None):
    """Performs bootstrap resampling on numpy arrays.
    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.
    Parameters
    ----------
    data : numpy.ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned
    Returns
    -------
    boot : numpy.ndarray
        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.
    Examples
    --------
    Obtain a twice resampled array:
    >>> from astropy.stats import bootstrap
    >>> import numpy as np
    >>> from astropy.utils import NumpyRNGContext
    >>> bootarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[6., 9., 0., 6., 1., 1., 2., 8., 7., 0.],
           [3., 5., 6., 3., 5., 3., 5., 8., 8., 0.]])
    >>> bootresult.shape
    (2, 10)
    Obtain a statistic on the array
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2, bootfunc=np.mean)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([4. , 4.6])
    Obtain a statistic with two outputs on the array
    >>> test_statistic = lambda x: (np.sum(x), np.mean(x))
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=test_statistic)
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[40. ,  4. ],
           [46. ,  4.6],
           [35. ,  3.5]])
    >>> bootresult.shape
    (3, 2)
    Obtain a statistic with two outputs on the array, keeping only the first
    output
    >>> bootfunc = lambda x:test_statistic(x)[0]
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=bootfunc)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([40., 46., 35.])
    >>> bootresult.shape
    (3,)
    """
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)
    
    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.
    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }
    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)
    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.
    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.
    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis."""

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
        
## make a parula map like in matlab
        
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
