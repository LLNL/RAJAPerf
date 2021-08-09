#!/usr/bin/env python
# coding: utf-8

# ##### Hatchet Notebook v0.1.0


# In[ ]:

import sys
import subprocess
import json
import os
import platform
import ipykernel

import pandas as pd
from IPython.display import display, HTML

machine = platform.uname().machine

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
        #generic_default_metric = gf.default_metric  # in newer Hatchet
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

def ExtractCommonSubtree(gf1: ht.GraphFrame,gf2: ht.GraphFrame) -> (ht.GraphFrame):
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
                s2 += common_subtree.dataframe.loc[nn,'time (inc)']
            elif nn._depth == 1:
                s1 += s2
                common_subtree.dataframe.loc[nn,'time (inc)'] = s2
                s2 = 0.0
            elif nn._depth == 0:
                common_subtree.dataframe.loc[nn,'time (inc)'] = s1
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
        gflist.append(ht.GraphFrame.from_caliper(CALI_FILES[i]['cali_file'], query))
        v  = gflist[i]
        vstrs.append(v.dataframe.iloc[0,v.dataframe.columns.get_loc('name')] + '_' + cstr)

    compare = pd.DataFrame(0,index=vstrs,columns=vstrs)

    for r in range(len(CALI_FILES)):
        for c in range(len(CALI_FILES)):
            r_common = ExtractCommonSubtree(GenericFrame(gflist[r]),GenericFrame(gflist[c]))
            c_common = ExtractCommonSubtree(GenericFrame(gflist[c]),GenericFrame(gflist[r]))
            scale = r_common / c_common
            df = scale.dataframe
            val = df.iloc[0, df.columns.get_loc('time (inc)')] 
            compare.loc[vstrs[r],vstrs[c]] = val
            print(".",end="")
    print(" ")
    print(compare)
    return(gflist)


# In[ ]:
# Add cali-query to PATH

#cali_query_path = "/usr/gapps/spot/live/caliper/" + machine + "/bin"
cali_query_path = "/home/skip/workspace/spack/opt/spack/linux-ubuntu20.04-haswell/gcc-10.2.0/caliper-2.5.0-y64d5flp5ph55dj74dpvaigjn62txxmc/bin"
os.environ["PATH"] += os.pathsep + cali_query_path
data_path="/home/skip/workspace/Caliper_test/testcommon/lassen_gcc831_cuda11/"

CALI_FILES = [ 
 { "cali_file": data_path+"RAJA_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"RAJA_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"RAJA_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Base_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Base_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Base_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Lambda_CUDA.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Lambda_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"Lambda_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
]

grouping_attribute = "prop:nested"
default_metric = "avg#inclusive#sum#time.duration" 
query = "select %s,sum(%s) group by %s format json-split" % (grouping_attribute, default_metric, grouping_attribute)


# In[ ]:
print("Comparing all variants to each other in new pandas dataframe")
gflist = CompareVariants(CALI_FILES)

#print(gflist[0].tree(metric_column="time (inc)"))
#display(HTML(gflist[0].dataframe.to_html()))

#(records,globals) = cr.read_caliper_contents(data_path+"Base_OpenMP.cali")
#print('\n')
#print(records)
#print('\n')
#print(globals)

Base_OpenMP = gflist[4]
Base_Seq = gflist[5]
Lambda_CUDA = gflist[6]

# In[ ]:
print("Illustrates two incompatible trees Base_OpenMP/Base_Seq")
#print(Base_OpenMP.tree(metric_column="time (inc)"))
#print(Lambda_CUDA.tree(metric_column="time (inc)"))
# compare Base_OpenMP to Base_Seq and display tree (illustrates incompatible trees)
invar =  Base_OpenMP / Base_Seq
print(invar.tree(metric_column="time (inc)"))

# In[ ]:
print("Illustrates a bit more compatible tree by fixing root nodes via GenericFrame ")
# compare  Base_OpenMP to Base_Seq and display tree (illustrates compatible trees by fixing Root Node)
outvar = GenericFrame(Base_OpenMP) / GenericFrame(Base_Seq)
print(outvar.tree(metric_column="time (inc)"))

# In[ ]:
# compare GenericFrame versions of Base_OpenMP to Lambda_CUDA and display tree (illustrates somewhat less compatible trees due to missing kernels)
print("Show variants that have a large difference in kernels run Base_OpenMP vs Lambda_CUDA")
cudavar = GenericFrame(Base_OpenMP) / GenericFrame(Lambda_CUDA)
print(cudavar.tree(metric_column="time (inc)"))

#cudavar2 = GenericFrame(Lambda_CUDA) / GenericFrame(Base_OpenMP)
#print(cudavar2.tree(metric_column="time (inc)"))

# In[ ]:
print("show use of ExtractCommonSubtree to fix variants with different kernels run")
common_subtree = ExtractCommonSubtree(GenericFrame(Base_OpenMP),GenericFrame(Lambda_CUDA))
print(common_subtree.tree(metric_column="time (inc)"))

common_subtree2 = ExtractCommonSubtree(GenericFrame(Lambda_CUDA),GenericFrame(Base_OpenMP))
print(common_subtree2.tree(metric_column="time (inc)"))

calc = common_subtree / common_subtree2
print(calc.tree(metric_column="time (inc)"))
display(HTML(calc.dataframe.to_html()))
# %%
