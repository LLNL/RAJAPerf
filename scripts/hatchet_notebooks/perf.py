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


import caliperreader as cr

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
        generic_default_metric = gf.default_metric
        generic_dataframe.iloc[0, generic_dataframe.columns.get_loc('name')] = 'Variant' 
        ii = generic_dataframe.index[0]
        fr = ht.frame.Frame({'name': 'Variant', 'type' : 'region'})
        nn = ht.node.Node(fr)
        setattr(nn,'_hatchet_nid',ii._hatchet_nid)
        setattr(nn,'_depth',ii._depth)
        setattr(nn,'children',ii.children)
        generic_dataframe.rename(index={ii: nn},inplace=True)
        setattr(generic_graph,'roots',[nn])
        super().__init__(generic_graph,generic_dataframe,generic_exc_metrics,generic_inc_metrics,generic_default_metric)

def CompareVariants(CALI_FILES):
    grouping_attribute = "prop:nested"
    default_metric = "avg#inclusive#sum#time.duration" 
    query = "select %s,sum(%s) group by %s format json-split" % (grouping_attribute, default_metric, grouping_attribute)
    gflist = []
    vstrs = []
    for i in range(len(CALI_FILES)):
        (records,globals) = cr.read_caliper_contents(CALI_FILES[i]['cali_file'])
        cstr = globals['compiler_suffix']
        gflist.append(ht.GraphFrame.from_caliper(CALI_FILES[i]['cali_file'], query))
        v  = gflist[i]
        vstrs.append(v.dataframe.iloc[0,v.dataframe.columns.get_loc('name')] + '_' + cstr)

    compare = pd.DataFrame(0,index=vstrs,columns=vstrs)

    for r in range(len(CALI_FILES)):
        for c in range(len(CALI_FILES)):
            rg = GenericFrame(gflist[r])
            cg = GenericFrame(gflist[c])
            scale = rg / cg
            df = scale.dataframe
            val = df.iloc[0, df.columns.get_loc('time (inc)')] 
            compare.loc[vstrs[r],vstrs[c]] = val

    print(compare)
    return(gflist)


# In[ ]:
# Add cali-query to PATH

#cali_query_path = "/usr/gapps/spot/live/caliper/" + machine + "/bin"
cali_query_path = "/home/skip/workspace/spack/opt/spack/linux-ubuntu20.04-haswell/gcc-10.2.0/caliper-2.5.0-y64d5flp5ph55dj74dpvaigjn62txxmc/bin"
os.environ["PATH"] += os.pathsep + cali_query_path
data_path="/home/skip/workspace/Caliper_test/by_variant/"

CALI_FILES = [ 
 { "cali_file":  data_path+"data_clang/Base_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file":  data_path+"data_clang/Base_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file":  data_path+"data_clang/Lambda_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file":  data_path+"data_clang/Lambda_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_clang/RAJA_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_clang/RAJA_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/Base_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/Base_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/Lambda_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/Lambda_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/RAJA_OpenMP.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
 { "cali_file": data_path+"data_gcc_10.2/RAJA_Seq.cali", "metric_name": "avg#inclusive#sum#time.duration"}, 
]

grouping_attribute = "prop:nested"
default_metric = "avg#inclusive#sum#time.duration" 
query = "select %s,sum(%s) group by %s format json-split" % (grouping_attribute, default_metric, grouping_attribute)


# In[ ]:
gflist = CompareVariants(CALI_FILES)

print(gflist[0].tree(metric_column="time (inc)"))
display(HTML(gflist[0].dataframe.to_html()))

(records,globals) = cr.read_caliper_contents(data_path+"data_clang/Base_OpenMP.cali")
#print('\n')
#print(records)
#print('\n')
print(globals)

# In[ ]:
# compare clang BASE_OpenMP to GCC BASE_OpenMP and display tree
invar = gflist[0] / gflist[6]
print(invar.tree(metric_column="time (inc)"))

# In[ ]:
# compare clang BASE_OpenMP to clang BASE_Seq and display tree (illustrates incompatible trees)
outvar = gflist[0] / gflist[1]
print(outvar.tree(metric_column="time (inc)"))

# In[ ]:
# compare GenericFrame versions of clang BASE_OpenMP to clang BASE_Seq and display tree (illustrates compatible trees)
genvar = GenericFrame(gflist[0]) / GenericFrame(gflist[1])
print(genvar.tree(metric_column="time (inc)"))

genvard = GenericFrame(gflist[0]) - GenericFrame(gflist[1])
print(genvard.tree(metric_column="time (inc)"))

# %%
