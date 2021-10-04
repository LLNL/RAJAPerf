#!/usr/bin/env python
# coding: utf-8

# ##### Hatchet Notebook v0.1.0


# In[ ]:

import sys
import subprocess
import json
from os import walk
from os.path import join
from collections import defaultdict
import platform
import ipykernel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
from IPython.display import display, HTML


machine = platform.uname().machine

use_native_reader=True
grouping_attribute = "prop:nested"
default_metric = "avg#inclusive#sum#time.duration" 
query = "select %s,sum(%s) group by %s format json-split" % (grouping_attribute, default_metric, grouping_attribute)


# Add hatchet to PYTHONPATH
deploy_dir = "/home/skip/workspace/hatchet/"
sys.path.append(deploy_dir)
sys.path.append(deploy_dir + "/hatchet")
import hatchet as ht 
#print(dir(ht))

# if you didn't install caliperreader via pip3 point to it's deploy_dir
# sys.path.append(deploy_dir + 'Caliper/python/caliper-reader/')
import caliperreader as cr # pip3 install caliper-reader

import importlib
import pkgutil

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    #print(results)
    return results

import_submodules(ht)

class GenericFrame(ht.GraphFrame):
    def __init__(self,gf):
        generic_dataframe=gf.dataframe.copy()
        generic_graph=gf.graph.copy()
        generic_exc_metrics = gf.exc_metrics
        generic_inc_metrics = gf.inc_metrics
        generic_default_metric = gf.default_metric  # in newer Hatchet
        #print('Default Metric = ' + gf.default_metric)
        generic_dataframe.iloc[0, generic_dataframe.columns.get_loc('name')] = 'Variant' 
        ii = generic_dataframe.index[0]
        #fr = ht.frame.Frame({'name': 'Variant', 'type' : 'region'})
        fr = ht.graphframe.Frame({'name': 'Variant', 'type' : 'region'})
        nn = ht.graphframe.Node(fr)
        setattr(nn,'_hatchet_nid',ii._hatchet_nid)
        setattr(nn,'_depth',ii._depth)
        setattr(nn,'children',ii.children)
        generic_dataframe.rename(index={ii: nn},inplace=True)
        setattr(generic_graph,'roots',[nn])
        super().__init__(generic_graph,generic_dataframe,generic_exc_metrics,generic_inc_metrics)

def ExtractCommonSubtree(gf1: ht.GraphFrame,gf2: ht.GraphFrame,metric:str) -> (ht.GraphFrame):
    if (gf1.graph == gf2.graph):
        return gf1
    else:
        cc = gf1.deepcopy()
        cc2 = gf2.deepcopy()
        cc.unify(cc2)
        # search for nodes contained in both graphs {0==both, 1==left only, 2==right only}
        filter_func = lambda x: x["_missing_node"] == 0
        common_subtree = cc.filter(filter_func, squash=True)
        #print(common_subtree.dataframe.columns.tolist())
        # tt is generator object from post order tree traversal, i.e starts down at first set of leaves
        tt = common_subtree.graph.roots[0].traverse(order="post")
        s2 = 0.0 # sum accumulated at depth 2
        s1 = 0.0 # sum accumulated at depth 1
        # replace subtree values with sum of kernels that have run
        for nn in tt:
            if nn._depth == 2:
                s2 += common_subtree.dataframe.loc[nn,metric]
            elif nn._depth == 1:
                s1 += s2
                common_subtree.dataframe.loc[nn,metric] = s2
                s2 = 0.0
            elif nn._depth == 0:
                common_subtree.dataframe.loc[nn,metric] = s1
                s1 = 0.0
    
        return common_subtree

def CompareVariants(CALI_FILES) -> (list):
    grouping_attribute = "prop:nested"
    default_metric = "avg#inclusive#sum#time.duration" 
    query = "select %s,sum(%s) group by %s format json-split" % (grouping_attribute, default_metric, grouping_attribute)

    gflist = []
    vstrs = []
    for i in range(len(CALI_FILES)):
        (records,globals) = cr.read_caliper_contents(CALI_FILES[i]['cali_file'])
        cstr = globals['compiler']
        if use_native_reader:
            gflist.append(ht.GraphFrame.from_caliperreader(CALI_FILES[i]['cali_file']))
        else:
            gflist.append(ht.GraphFrame.from_caliper(CALI_FILES[i]['cali_file'], query))
        v  = gflist[i]
        vstrs.append(v.dataframe.iloc[0,v.dataframe.columns.get_loc('name')] + '_' + cstr)

    compare = pd.DataFrame(0,index=vstrs,columns=vstrs)

    for r in range(len(CALI_FILES)):
        for c in range(len(CALI_FILES)):
            r_common = ExtractCommonSubtree(GenericFrame(gflist[r]),GenericFrame(gflist[c]),default_metric)
            c_common = ExtractCommonSubtree(GenericFrame(gflist[c]),GenericFrame(gflist[r]),default_metric)
            scale = r_common / c_common
            df = scale.dataframe
            if use_native_reader:
                val = df.iloc[0, df.columns.get_loc(default_metric)] 
            else:
                val = df.iloc[0, df.columns.get_loc('time (inc)')] 
            compare.loc[vstrs[r],vstrs[c]] = val
            print(".",end="")
    print(" ")
    print(compare)
    return(gflist)

def ReadCali(directory) -> (list):
    CALI_FILES = []
    allfiles = [join(root,f) for root,dirs,files in os.walk(directory) for f in files if f.endswith('.cali')]
    for f in allfiles:
        dd = {'cali_file': f , 'metric_name' : 'avg#inclusive#sum#time.duration'}
        CALI_FILES.append(dd)
    return CALI_FILES
    
def ExtractKernelList(CALI_FILES,variants):
    kernel_list=[]
    candidate_list=[]
    for i in range(len(CALI_FILES)):
        (records,globals) = cr.read_caliper_contents(CALI_FILES[i]['cali_file'])
        variant = globals['variant']
        if variant in variants:
            gf = ht.GraphFrame.from_caliperreader(CALI_FILES[i]['cali_file'])
            tt = gf.graph.roots[0].traverse(order="post")
            for nn in tt:
                if nn._depth == 2:
                    candidate_list.append(gf.dataframe.loc[nn,'name'])
            kernel_list = list(set(candidate_list) | set(kernel_list))
    return kernel_list

def PlotSweep(CALI_FILES,kernel,variants):
    plt.title(kernel)
    vdata = defaultdict(list)
    for i in range(len(CALI_FILES)):
        (records,globals) = cr.read_caliper_contents(CALI_FILES[i]['cali_file'])
        psize = float(globals['ProblemSize'])
        variant = globals['variant']
        cluster = globals['cluster']
        if variant in variants:
            gf = ht.GraphFrame.from_caliperreader(CALI_FILES[i]['cali_file'])
            df = gf.dataframe
            metric = CALI_FILES[i]['metric_name']
            val = 0.0
            try:
                val = df.loc[df['name']==kernel].iloc[0][metric]
            except: pass
            vdata[cluster+'_'+variant].append((psize,val))
    legend = []    
    for k,v in vdata.items():
        legend.append(k)
        ll = sorted(v,key = lambda x: x[0])
        x = [num[0] for num in ll]
        y = [num[1] for num in ll]
        plt.plot(x,y)
    plt.xlabel('problem size')
    plt.ylabel('time')
    plt.xscale('log',base=2)
    plt.yscale('log',base=2)
    plt.legend(legend)


        

# In[ ]:
# Add cali-query to PATH

#cali_query_path = "/usr/gapps/spot/live/caliper/" + machine + "/bin"
cali_query_path = "/home/skip/workspace/spack/opt/spack/linux-ubuntu20.04-haswell/gcc-10.2.0/caliper-2.5.0-y64d5flp5ph55dj74dpvaigjn62txxmc/bin"
os.environ["PATH"] += os.pathsep + cali_query_path
data_path="/home/skip/cali_sweep/RAJAPerf_0"

CALI_FILES = ReadCali(data_path)
#print(CALI_FILES)

#CALI_FILES = [ 
# { "cali_file": data_path+"RAJA_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"RAJA_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"RAJA_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Base_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Base_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Base_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Lambda_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Lambda_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
# { "cali_file": data_path+"Lambda_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
#]

# In[]:
pp = PdfPages('RAJAPERF_Sweep.pdf')
variants = ['Lambda_OpenMP','Base_OpenMP','RAJA_OpenMP']
klist = sorted(ExtractKernelList(CALI_FILES,variants))
fignum = 0
for i in range(len(klist)):
    fig = plt.figure(fignum)
    PlotSweep(CALI_FILES,klist[i],variants)
    plt.show()
    pp.savefig(fig)
    fignum += 1
pp.close()


# In[ ]:
if use_native_reader:
    # New technique
    gf0=GenericFrame(ht.GraphFrame.from_caliperreader(CALI_FILES[0]['cali_file']))
    display(HTML(gf0.dataframe.to_html()))

    (records,globals) = cr.read_caliper_contents(CALI_FILES[0]['cali_file'])
    print(globals)

    print("New technique: ")
    print(gf0.show_metric_columns())
    print(gf0.tree(metric_column="avg#inclusive#sum#time.duration"))
    gf1=GenericFrame(ht.GraphFrame.from_caliperreader(CALI_FILES[1]['cali_file']))
    print(gf1.tree(metric_column="avg#inclusive#sum#time.duration"))
    outvar=gf0/gf1
    print(outvar.tree(metric_column="avg#inclusive#sum#time.duration"))

# In[ ]:
if not use_native_reader:
    #Old technique
    gf1 = ht.GraphFrame.from_caliper(CALI_FILES[2]['cali_file'], query)
    display(HTML(gf1.dataframe.to_html()))
    print("Old technique: ")
    print(gf1.show_metric_columns())
    print(GenericFrame(gf1).tree(metric_column='time (inc)'))


# In[ ]:
print("Comparing all variants to each other in new pandas dataframe")
gflist = CompareVariants(CALI_FILES[0:6])

#print(gflist[0].tree(metric_column="time (inc)"))
#display(HTML(gflist[0].dataframe.to_html()))

#(records,globals) = cr.read_caliper_contents(data_path+"Base_OpenMP.cali")
#print('\n')
#print(records)
#print('\n')
#print(globals)

#Base_OpenMP = gflist[4]
#Base_Seq = gflist[5]
#Lambda_CUDA = gflist[6]


# %%
