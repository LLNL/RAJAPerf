#!/usr/bin/env python3

import sys
import platform
import datetime as dt

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-r','--report',required=True,nargs=1,help="Pass the Caliper report file.")
parser.add_argument('-b','--baseline',required=True,nargs=1,help="Pass the Caliper baseline file.")
parser.add_argument('-t','--tolerance',required=False,nargs=1,type=float,default=[0.05],help="Specify tolerance for pass/fail")

args = parser.parse_args()
print(args)

input_deploy_dir_str = "/usr/gapps/spot/dev"
machine = platform.uname().machine

sys.path.append(input_deploy_dir_str + "/hatchet-venv/" + machine + "/lib/python3.7/site-packages")
sys.path.append(input_deploy_dir_str + "/hatchet/" + machine)
sys.path.append(input_deploy_dir_str + "/spotdb")

import hatchet as ht

# This class turns an existing GraphFrame into a "generic" one by renaming
# the root node into a generic node. We can then compare 2 "generic" graph
# frame. In practice we use it to allow Hatchet to compare performance trees
# generated from RAJA and Base kernels.
# If they don’t have exactly the same structure, then we can use
# ExtractCommonSubtree below.
class GenericFrame(ht.GraphFrame):
   def __init__(self, gf):
      generic_dataframe = gf.dataframe.copy()
      generic_graph = gf.graph.copy()
      generic_exc_metrics = gf.exc_metrics
      generic_inc_metrics = gf.inc_metrics
      generic_default_metric = gf.default_metric  # in newer Hatchet
      generic_dataframe.iloc[0, generic_dataframe.columns.get_loc('name')] = 'Variant'
      ii = generic_dataframe.index[0]
      # fr = ht.frame.Frame({'name': 'Variant', 'type' : 'region'})
      fr = ht.graphframe.Frame({'name': 'Variant', 'type': 'region'})
      nn = ht.graphframe.Node(fr)
      setattr(nn, '_hatchet_nid', ii._hatchet_nid)
      setattr(nn, '_depth', ii._depth)
      setattr(nn, 'children', ii.children)
      generic_dataframe.rename(index={ii: nn}, inplace=True)
      setattr(generic_graph, 'roots', [nn])
      super().__init__(generic_graph, generic_dataframe, generic_exc_metrics, generic_inc_metrics)

# In this class, we turn dissimilar GraphFrames into comparable ones.
# The idea behind is that the trees contain timings for the same algorithms
# but different implementations (tuning) that result in non comparable leaves.
# We extract the minimal value of the lowest level data leaves to set
# a common comparison dataset.
# To understand the implementation below, note that the caliper annotation
# follows a 3-level structure:
# Variant
# - Group
# -- Kernel
# --- Kernel.Tuning
def ExtractCommonSubtree(gf1: ht.GraphFrame, gf2: ht.GraphFrame, metric: str) -> (ht.GraphFrame):
   if (gf1.graph == gf2.graph):
      return gf1
   else:
      cc = gf1.deepcopy()
      cc2 = gf2.deepcopy()
      cc.unify(cc2)
      # search for nodes contained in both graphs {0==both, 1==left only, 2==right only}
      filter_func = lambda x: x["_missing_node"] == 0
      common_subtree = cc.filter(filter_func, squash=True)
      # print(common_subtree.dataframe.columns.tolist())
      # tt is generator object from post order tree traversal, i.e starts down at first set of leaves
      tt = common_subtree.graph.roots[0].traverse(order="post")
      s2 = 0.0  # sum accumulated at depth 2
      s1 = 0.0  # sum accumulated at depth 1
      s0 = 0.0
      m3 = sys.float_info.max
      # replace subtree values with sum of kernels that have run
      for nn in tt:
         if nn._depth == 3:
            if common_subtree.dataframe.loc[nn, metric] < m3:
               m3 = common_subtree.dataframe.loc[nn, metric]
         elif nn._depth == 2:
            s2 = m3
            s1 += s2
            common_subtree.dataframe.loc[nn, metric] = s2
            m3 = sys.float_info.max
            s2 = 0
         elif nn._depth == 1:
            s0 += s1
            common_subtree.dataframe.loc[nn, metric] = s1
            s1 = 0
         elif nn._depth == 0:
            common_subtree.dataframe.loc[nn, metric] = s0

      return common_subtree

f1 = args.report[0]
f2 = args.baseline[0]
tolerance=args.tolerance[0]


gf1 = GenericFrame(ht.GraphFrame.from_caliperreader(f1))
gf2 = GenericFrame(ht.GraphFrame.from_caliperreader(f2))

if 'min#inclusive#sum#time.duration' in gf1.inc_metrics:
  metric = 'min#inclusive#sum#time.duration' 
elif 'Min time/rank' in gf1.inc_metrics:
  metric = 'Min time/rank' 

gf11 = gf1
gf22 = gf2

if len(gf1.graph) != len(gf2.graph):
   gf11 = ExtractCommonSubtree(gf1,gf2,metric)
   gf22 = ExtractCommonSubtree(gf2,gf1,metric)

gf3 = gf11 - gf22

# Sort resulting DataFrame by ``time`` column in descending order.
#sorted_df = gf3.dataframe.sort_values(by=[metric], ascending=False)

# Display resulting DataFrame.
#print(sorted_df.head())

# Display calltree
#print(gf3.tree(metric_column=metric,precision=5))

#setup threshold as a fraction of baseline using tolerance multiplier
baseline_node = gf2.graph.roots[0]
threshold = tolerance * float(gf2.dataframe.loc[baseline_node, metric])


# Get a single metric value for a given node
root_node = gf3.graph.roots[0]
result = gf3.dataframe.loc[root_node, metric]
print("Result =", result," with threshold =",threshold)
if result > threshold:
  print('fail')
else:
  print('pass')

