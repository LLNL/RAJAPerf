#!/usr/bin/env python3

import sys
import platform
import datetime as dt

from optparse import OptionParser

parser=optparse.OptionParser()

parser.add_option("-r", "--report", metavar="FILE", help="Pass the caliper report file.")
parser.add_option("-b", "--baseline", metavar="FILE", help="Pass the caliper baseline file.")

(options, args) = parser.parse_args()

input_deploy_dir_str = "/usr/gapps/spot/dev"
machine = platform.uname().machine

sys.path.append(input_deploy_dir_str + "/hatchet-venv/" + machine + "/lib/python3.7/site-packages")
sys.path.append(input_deploy_dir_str + "/hatchet/" + machine)
sys.path.append(input_deploy_dir_str + "/spotdb")

import hatchet as ht

f1 = options.report
f2 = options.baseline

gf1 = ht.GraphFrame.from_caliperreader(f1)
gf2 = ht.GraphFrame.from_caliperreader(f2)

gf3 = gf2 - gf1

# Display dataframe columns
print(gf3.dataframe.columns)

# Sort resulting DataFrame by ``time`` column in descending order.
sorted_df = gf3.dataframe.sort_values(by=["sum#sum#time.duration"], ascending=False)

# Display resulting DataFrame.
print(sorted_df.head())

# Display calltree
print(gf3.tree(metric_column="sum#sum#time.duration"))

# Count number of nodes in calltree
print("Num nodes:", len(gf3.graph))

# Get a single metric value for a given node
root_node = gf3.graph.roots[0]
metric = "sum#sum#time.duration"
print("")
print("Node name =", root_node.frame["name"])
print(metric, "=", gf3.dataframe.loc[root_node, metric])
