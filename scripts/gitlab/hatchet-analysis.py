#!/usr/bin/env python3

import sys
import platform
import datetime as dt

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-r", "--report", metavar="FILE", help="Pass the caliper report file.")
parser.add_option("-b", "--baseline", metavar="FILE", help="Pass the caliper baseline file.")

(options, args) = parser.parse_args()

#input_deploy_dir_str = "/usr/gapps/spot/dev"
#machine = platform.uname().machine

#sys.path.append(input_deploy_dir_str + "/hatchet-venv/" + machine + "/lib/python3.7/site-packages")
#sys.path.append(input_deploy_dir_str + "/hatchet/" + machine)
#sys.path.append(input_deploy_dir_str + "/spotdb")

import hatchet as ht

class GenericFrame(ht.GraphFrame):
   def __init__(self, gf):
      generic_dataframe = gf.dataframe.copy()
      generic_graph = gf.graph.copy()
      generic_exc_metrics = gf.exc_metrics
      generic_inc_metrics = gf.inc_metrics
      generic_default_metric = gf.default_metric  # in newer Hatchet
      print('Default Metric = ' + gf.default_metric)
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
         #print(nn._depth)
         if nn._depth == 3:
            if common_subtree.dataframe.loc[nn, metric] < m3:
               m3 = common_subtree.dataframe.loc[nn, metric]
               #print(m3)
         elif nn._depth == 2:
            s2 = m3
            s1 += s2
            common_subtree.dataframe.loc[nn, metric] = s2
            m3 = sys.float_info.max
            #print(s2)
            s2 = 0
         elif nn._depth == 1:
            s0 += s1
            common_subtree.dataframe.loc[nn, metric] = s1
            #print(s1)
            s1 = 0
         elif nn._depth == 0:
            common_subtree.dataframe.loc[nn, metric] = s0
            #print(s0)

      
      return common_subtree

f1 = options.report
f2 = options.baseline

#metric = "sum#inclusive#sum#time.duration"
metric = "Min time/rank"

gf1 = GenericFrame(ht.GraphFrame.from_caliperreader(f1))
gf2 = GenericFrame(ht.GraphFrame.from_caliperreader(f2))

print("Num nodes gf1:", len(gf1.graph))
print("Num nodes gf2:", len(gf2.graph))
if len(gf1.graph) != len(gf2.graph):
   gf1c = ExtractCommonSubtree(gf1,gf2,metric)
   gf2c = ExtractCommonSubtree(gf2,gf1,metric)
   gf3 = gf1c - gf2c
else:
   gf3 = gf1 - gf2

# Display dataframe columns
print(gf3.dataframe.columns)

# Sort resulting DataFrame by ``time`` column in descending order.
sorted_df = gf3.dataframe.sort_values(by=[metric], ascending=False)

# Display resulting DataFrame.
print(sorted_df.head())

# Display calltree
print(gf1c.tree(metric_column=metric,precision=5))
print(gf2c.tree(metric_column=metric,precision=5))
print(gf3.tree(metric_column=metric,precision=5))

# Count number of nodes in calltree
print("Num nodes:", len(gf3.graph))

# Get a single metric value for a given node
root_node = gf3.graph.roots[0]
print("")
print("Node name =", root_node.frame["name"])
print(metric, "=", gf3.dataframe.loc[root_node, metric])
