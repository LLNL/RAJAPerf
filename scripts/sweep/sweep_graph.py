#!/usr/bin/env python3
import math
import os
import sys
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse_sweep_graph as ac
import data_classes_sweep_graph as dc
def read_runinfo_file(sweep_index, sweep_subdir_runinfo_file_path, run_size_index):
   #print("read_runinfo_file")
   #print(sweep_index, sweep_subdir_runinfo_file_path, run_size_index)
   with open(sweep_subdir_runinfo_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      ignore = True
      c_to_info_kinds = {}
      for row in file_reader:
         # print(row)
         if row[0].strip() == "Kernels":
            ignore = False
            for c in range(1, len(row)):
               info_kind = row[c].strip()
               #print(c, info_kind)
               if not info_kind in dc.Data.kinds:
                  # add new kind to global data
                  print("Unknown kernel info {0}".format(info_kind))
                  dc.Data.kinds[info_kind] = dc.Data.DataTree(info_kind, "info", dc.Data.info_axes)
               if info_kind in c_to_info_kinds:
                  print("Repeated kernel info {0}".format(info_kind))
                  sys.exit(1)
               if not dc.Data.kinds[info_kind].data:
                  #print("# add data to kind:" + info_kind)
                  dc.Data.kinds[info_kind].makeData()
               if not sweep_index in dc.Data.kinds[info_kind].data.data:
                  #print("# add new sweep to global data")
                  dc.Data.kinds[info_kind].data.data[sweep_index] = {}
               if run_size_index in dc.Data.kinds[info_kind].data.data[sweep_index]:
                  sweep_dir_name = dc.Data.get_index_name(dc.Data.axes["sweep_dir_name"], sweep_index)
                  run_size_name = dc.Data.get_index_name(dc.Data.axes["run_size"], run_size_index)
                  print("Repeated kernel size {0} in {1}".format(sweep_dir_name, run_size_name))
                  sys.exit(1)
               else:
                  #print("# add new size to global data")
                  dc.Data.kinds[info_kind].data.data[sweep_index][run_size_index] = {}
               # make map of columns to names
               c_to_info_kinds[c] = info_kind
               c_to_info_kinds[info_kind] = c
         elif not ignore:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in dc.Data.kernels:
               kernel_index = dc.Data.kernels[kernel_name]
            elif (len(dc.Data.include_kernels) == 0 or kernel_name in dc.Data.include_kernels) and (not kernel_name in dc.Data.exclude_kernels):
               # add kernel to global list
               dc.Data.add_kernel(kernel_name)
               kernel_index = dc.Data.kernels[kernel_name]
            else:
               continue # skip this kernel

            for c in range(1, len(row)):
               info_kind = c_to_info_kinds[c]
               try:
                  # add data to global structure
                  val = int(row[c].strip())
                  #print(kernel_index, kernel_name, info_kind, val)

                  axes_index = { dc.Data.axes["sweep_dir_name"]: sweep_index,
                                 dc.Data.axes["run_size"]: run_size_index,
                                 dc.Data.axes["kernel_index"]: kernel_index, }

                  dc.Data.kinds[info_kind].set(axes_index, val)
               except ValueError:
                  print('read_runinfo_file ValueError')
                  pass # could not convert data to int

# we expect the following to overlap wrt redundancies to read_caliper_timing_file; they should be refactored
def read_caliper_runinfo_file(cr, sweep_index, sweep_subdir, run_size_index):
   #print(sweep_index, sweep_subdir, run_size_index)
   kernel_list = []
   candidate_list = []
   kernel_metadata = {}

   # per kernel metadata is in dataframe columns per kernel.tuning, so we need our kernel list
   allfiles = sorted(glob.glob(glob.escape(sweep_subdir) + "/*.cali"))
   # not all kernels run in every variant so capture kernel list across variants
   for f in allfiles:
      #print(f)
      gf = cr.GraphFrame.from_caliperreader(f)
      metric = 'min#inclusive#sum#time.duration'
      #print(gf.inc_metrics)

      # extract kernel list
      kernel_index = -1
      tt = gf.graph.roots[0].traverse(order="pre")
      for nn in tt:
         # test if leaf node
         if not nn.children:
            # kernel_tuning_name is kernel.tuning in Caliper
            kernel_tuning_name = gf.dataframe.loc[nn, 'name']
            kernel_name = kernel_tuning_name.split('.')[0]
            if kernel_name not in kernel_metadata:
               kernel_metadata[kernel_name] = {}
            if (len(dc.Data.include_kernels) == 0 or kernel_name in dc.Data.include_kernels) and (not kernel_name in dc.Data.exclude_kernels):
               candidate_list.append(kernel_name)
               if len(kernel_metadata[kernel_name]) == 0:
                  metadata = {}
                  metadata['Problem size'] = gf.dataframe.loc[nn, 'any#any#max#ProblemSize']
                  metadata['Reps'] = gf.dataframe.loc[nn, 'any#any#max#Reps']
                  metadata['Iterations/rep'] = gf.dataframe.loc[nn, 'any#any#max#Iterations/Rep']
                  metadata['Kernels/rep'] = gf.dataframe.loc[nn, 'any#any#max#Kernels/Rep']
                  metadata['Bytes/rep'] = gf.dataframe.loc[nn, 'any#any#max#Bytes/Rep']
                  metadata['FLOPS/rep'] = gf.dataframe.loc[nn, 'any#any#max#Flops/Rep']
                  kernel_metadata[kernel_name] = metadata
                  #print("Kernel Column Metadata:" + kernel_name )
                  #print(kernel_metadata[kernel_name]['Problem size'])
               
      kernel_list = list(set(candidate_list) | set(kernel_list))
  
   for kernel_name in kernel_list:
      if kernel_name not in dc.Data.kernels:
         dc.Data.add_kernel(kernel_name)
      kernel_index = dc.Data.kernels[kernel_name]
      metadata = kernel_metadata[kernel_name] # use column metadata instead
      for info_kind, info_value in metadata.items():
         if not info_kind in dc.Data.kinds:
            dc.Data.kinds[info_kind] = dc.Data.DataTree(info_kind, "info", dc.Data.info_axes)
         if not dc.Data.kinds[info_kind].data:
            dc.Data.kinds[info_kind].makeData()
         if not sweep_index in dc.Data.kinds[info_kind].data.data:
            dc.Data.kinds[info_kind].data.data[sweep_index] = {}
         if not run_size_index in dc.Data.kinds[info_kind].data.data[sweep_index]:
            dc.Data.kinds[info_kind].data.data[sweep_index][run_size_index] = {}
         try:
            val = int(info_value)
            axes_index = { dc.Data.axes["sweep_dir_name"]: sweep_index,
                           dc.Data.axes["run_size"]: run_size_index,
                           dc.Data.axes["kernel_index"]: kernel_index, }
            dc.Data.kinds[info_kind].set(axes_index, val)
            #sweep_dir_name = dc.Data.get_index_name(dc.Data.axes["sweep_dir_name"], sweep_index)
            #run_size_name = dc.Data.get_index_name(dc.Data.axes["run_size"], run_size_index)
            #kernel_index_name = dc.Data.get_index_name(dc.Data.axes["kernel_index"], kernel_index)
            #print("Info kind {0} {1} size {2} kernel {3} val {4}".format(info_kind,sweep_dir_name, run_size_name,kernel_index_name,val))
         except ValueError:
            print("read_caliper_runinfo_file ValueError")
            pass # could not convert data to int

def read_timing_file(sweep_index, sweep_subdir_timing_file_path, run_size_index):
   #print(sweep_index, sweep_subdir_timing_file_path, run_size_index)
   with open(sweep_subdir_timing_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      data_kind = dc.g_timing_file_kind
      if not data_kind in dc.Data.kinds:
         raise NameError("Unknown kind {}".format(data_kind))
      if not dc.Data.kinds[data_kind].data:
         dc.Data.kinds[data_kind].makeData()
      if not sweep_index in dc.Data.kinds[data_kind].data.data:
         dc.Data.kinds[data_kind].data.data[sweep_index] = {}
      if not run_size_index in dc.Data.kinds[data_kind].data.data[sweep_index]:
         dc.Data.kinds[data_kind].data.data[sweep_index][run_size_index] = {}
      else:
         sweep_dir_name = dc.Data.get_index_name(dc.Data.axes["sweep_dir_name"], sweep_index)
         run_size_name = dc.Data.get_index_name(dc.Data.axes["run_size"], run_size_index)
         raise NameError("Already seen {0} in {1}".format(sweep_dir_name, run_size_name))

      c_to_variant_index = {}
      c_to_tuning_index = {}
      for row in file_reader:
         # print(row)
         if row[0].strip() == "Kernel":
            if len(c_to_variant_index) == 0:
               for c in range(1, len(row)):
                  variant_name = row[c].strip()
                  variant_index = -1
                  if variant_name in dc.Data.variants:
                     variant_index = dc.Data.variants[variant_name]
                  elif (len(dc.Data.include_variants) == 0 or variant_name in dc.Data.include_variants) and (not variant_name in dc.Data.exclude_variants):
                     dc.Data.add_variant(variant_name)
                     variant_index = dc.Data.variants[variant_name]
                  else:
                     variant_index = -1
                  c_to_variant_index[c] = variant_index

            elif len(c_to_tuning_index) == 0:
               for c in range(1, len(row)):
                  tuning_name = row[c].strip()
                  if tuning_name in dc.Data.tunings:
                     tuning_index = dc.Data.tunings[tuning_name]
                  elif (len(dc.Data.include_tunings) == 0 or tuning_name in dc.Data.include_tunings) and (not tuning_name in dc.Data.exclude_tunings):
                     dc.Data.add_tuning(tuning_name)
                     tuning_index = dc.Data.tunings[tuning_name]
                  else:
                     tuning_index = -1
                  c_to_tuning_index[c] = tuning_index

            else:
               print("Unknown row {0}".format(row))
               sys.exit(1)
         elif len(c_to_variant_index) > 0 and len(c_to_tuning_index) > 0:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in dc.Data.kernels:
               kernel_index = dc.Data.kernels[kernel_name]
            else:
               continue # skip kernel

            for c in range(1, len(row)):
               variant_index = c_to_variant_index[c]
               tuning_index = c_to_tuning_index[c]
               if variant_index < 0 or tuning_index < 0:
                  continue # ignore data

               axes_index = { dc.Data.axes["sweep_dir_name"]: sweep_index,
                              dc.Data.axes["run_size"]: run_size_index,
                              dc.Data.axes["kernel_index"]: kernel_index,
                              dc.Data.axes["variant_index"]: variant_index,
                              dc.Data.axes["tuning_index"]: tuning_index, }
               #print(axes_index)
               #print(dc.Data.axes)
               try:
                  val = float(row[c].strip())
                  #print(kernel_index, kernel_name, variant_index, tuning_index, data_kind, val)
                  dc.Data.kinds[data_kind].set(axes_index, val)
               except ValueError:
                  # we usually encounter this for Not run entry
                  #print(row[c].strip())
                  #print('read_timing_file ValueError')
                  pass # could not convert data to float

def read_caliper_timing_file(cr, sweep_index, sweep_subdir, run_size_index):
   graph_frames = []
   kernel_list = []
   candidate_list = []

   data_kind = dc.g_timing_file_kind
   if not data_kind in dc.Data.kinds:
      raise NameError("Unknown kind {}".format(data_kind))
   if not dc.Data.kinds[data_kind].data:
      dc.Data.kinds[data_kind].makeData()
   if not sweep_index in dc.Data.kinds[data_kind].data.data:
      dc.Data.kinds[data_kind].data.data[sweep_index] = {}
   if not run_size_index in dc.Data.kinds[data_kind].data.data[sweep_index]:
      dc.Data.kinds[data_kind].data.data[sweep_index][run_size_index] = {}
   else:
      sweep_dir_name = dc.Data.get_index_name(dc.Data.axes["sweep_dir_name"], sweep_index)
      run_size_name = dc.Data.get_index_name(dc.Data.axes["run_size"], run_size_index)
      raise NameError("Already seen {0} in {1}".format(sweep_dir_name, run_size_name))

   #print("run size:" + Data.get_index_name(Data.axes["run_size"], run_size_index))
   allfiles = sorted(glob.glob(glob.escape(sweep_subdir) + "/*.cali"))
   for f in allfiles:
      kernel_tuning_list = []
      candidate_tuning_list = []
      gf = cr.GraphFrame.from_caliperreader(f)
      #print(gf.metadata['variant'])
      metric = 'min#inclusive#sum#time.duration'
      #print(gf.inc_metrics)
      graph_frames.append(gf)

      #take care of variant in this graphframe
      variant_name = gf.metadata['variant']
      
      if variant_name in dc.Data.variants:
         variant_index = dc.Data.variants[variant_name]
      elif (len(dc.Data.include_variants) == 0 or variant_name in dc.Data.include_variants) and (not variant_name in dc.Data.exclude_variants):
         dc.Data.add_variant(variant_name)
         variant_index = dc.Data.variants[variant_name]
      else:
         variant_index = -1

      # extract kernel list
      kernel_index = -1
      tt = gf.graph.roots[0].traverse(order="pre")
      for nn in tt:
         # test if leaf node
         if not nn.children:
            #kernel_tuning_name is kernel.tuning in Caliper
            kernel_tuning_name = gf.dataframe.loc[nn,'name']
            kernel_name = kernel_tuning_name.split('.')[0]
            if kernel_name in dc.Data.kernels:
               kernel_tuning_name = gf.dataframe.loc[nn,'name']
               candidate_tuning_list.append(kernel_tuning_name)
               candidate_list.append(kernel_name)
      kernel_list = list(set(candidate_list) | set(kernel_list))
      kernel_tuning_list = list(set(candidate_tuning_list) | set(kernel_tuning_list))
      #print(kernel_list)
      #print(kernel_tuning_list)

      for kernel in kernel_tuning_list:
         kernel_name = kernel.split('.')[0]
         tuning_name = kernel.split('.')[1]

         if kernel_name in dc.Data.kernels:
            kernel_index = dc.Data.kernels[kernel_name]
         else:
            continue # skip kernel

         if tuning_name in dc.Data.tunings:
            tuning_index = dc.Data.tunings[tuning_name]
         elif (len(dc.Data.include_tunings) == 0 or tuning_name in dc.Data.include_tunings) and (not tuning_name in dc.Data.exclude_tunings):
            dc.Data.add_tuning(tuning_name)
            tuning_index = dc.Data.tunings[tuning_name]
         else:
            tuning_index = -1
            
         if variant_index < 0 or tuning_index <0:
            continue # skip this variant or tuning

         axes_index = { dc.Data.axes["sweep_dir_name"]: sweep_index,
                        dc.Data.axes["run_size"]: run_size_index,
                        dc.Data.axes["kernel_index"]: kernel_index,
                        dc.Data.axes["variant_index"]: variant_index,
                        dc.Data.axes["tuning_index"]: tuning_index, }
         val = 0.0
         #print(metric)
         try:
            val = float(gf.dataframe.loc[gf.dataframe['name']==kernel].iloc[0][metric])
            #print(variant_name, kernel_name, tuning_name, data_kind, val)
            dc.Data.kinds[data_kind].set(axes_index, val)
         except ValueError:
            print('ValueError')
            pass # could not convert data to float

def get_plot_data(kind, partial_axes_index):

   if not kind in dc.Data.kinds:
      raise NameError("Unknown kind {}".format(kind))

   kind_data = dc.Data.kinds[kind]

   assert(kind_data.hasAxes(partial_axes_index))

   data = []
   for axes_index, leftover_axes_index, value in kind_data.partial_match_items(partial_axes_index):
      index_name = kind_data.indexName(leftover_axes_index)
      data.append({ "name": index_name,
                    "axes_index": leftover_axes_index,
                    "data": [value] })

   return data

def get_plot_data2(xkind, ykind, partial_axes_index):

   if not ykind in dc.Data.kinds:
      raise NameError("Unknown kind {}".format(ykind))
   if not xkind in dc.Data.kinds:
      raise NameError("Unknown kind {}".format(xkind))

   ykind_data = dc.Data.kinds[ykind]
   xkind_data = dc.Data.kinds[xkind]

   assert(ykind_data.hasAxes(partial_axes_index))
   assert(xkind_data.hasAxes(partial_axes_index))

   data = []
   for axes_index, leftover_axes_index, yvalue in ykind_data.partial_match_items(partial_axes_index):
      index_name = ykind_data.indexName(leftover_axes_index)
      xvalue = xkind_data.get(axes_index)
      data.append({ "name": index_name,
                    "axes_index": leftover_axes_index,
                    "ydata": [yvalue],
                    "xdata": [xvalue] })

   return data

g_gname = None

g_lloc = 'best'

g_ylabel = None
g_yscale = None
g_ylim = None

g_xlabel = None
g_xscale = None
g_xlim = None

g_hbin_size = None

def plot_data_split_line(outputfile_name, split_axis_name, xaxis_name, xkind, ykinds):
   print("plotting {} {} {} {}".format(outputfile_name, split_axis_name, xaxis_name, xkind, ykinds))

   assert(split_axis_name == "kernel_index")
   for split_index in range(0, dc.Data.num_kernels):
      split_name = dc.Data.kernels[split_index]

      lloc = g_lloc

      ylabel = g_ylabel
      yscale = g_yscale or "log"
      ylim = g_ylim

      xlabel = g_xlabel or dc.Data.kinds[xkind].label
      xscale = g_xscale or "log"
      xlim = g_xlim

      gname = g_gname

      split_data = { "ynames": [],
                     "ycolor": {},
                     "yformat": {},
                     "ydata": {},
                     "xdata": {}, }

      for ykind in ykinds:
         if gname:
            gname = "{}\n{}".format(gname, ykind)
         else:
            gname = "{}".format(ykind)
         if not ykind in dc.Data.kinds:
            raise NameError("Unknown kind {}".format(ykind))
         if not ylabel:
            ylabel = dc.Data.kinds[ykind].label
         elif (not g_ylabel) and ylabel != dc.Data.kinds[ykind].label:
            raise NameError("kinds use different labels {}".format([dc.Data.kinds[_ykind].label for _ykind in ykinds]))

         assert(xaxis_name == "run_size")
         for x_index in range(0, dc.Data.num_run_sizes):

            axes_index = { dc.Data.axes[split_axis_name]: split_index,
                           dc.Data.axes[xaxis_name]: x_index  }

            data_list = get_plot_data2(xkind, ykind, axes_index)

            for data in data_list:
               yname = data["name"]

               if not yname in split_data["ydata"]:

                  ycolor = (0.0, 0.0, 0.0, 1.0)
                  if dc.Data.axes["variant_index"] in data["axes_index"]:
                     variant_index = data["axes_index"][dc.Data.axes["variant_index"]]
                     ycolor = dc.Data.variant_colors[variant_index]

                  ymarker = ""
                  if dc.Data.axes["sweep_dir_name"] in data["axes_index"]:
                     sweep_index = data["axes_index"][dc.Data.axes["sweep_dir_name"]]
                     ymarker = dc.Data.sweep_markers[sweep_index]

                  yformat = "{}-".format(ymarker)
                  if dc.Data.axes["tuning_index"] in data["axes_index"]:
                     tuning_index = data["axes_index"][dc.Data.axes["tuning_index"]]
                     yformat = "{}{}".format(ymarker, dc.Data.tuning_formats[tuning_index])

                  split_data["ynames"].append(yname)
                  split_data["ycolor"][yname] = ycolor
                  split_data["yformat"][yname] = yformat
                  split_data["ydata"][yname] = []
                  split_data["xdata"][yname] = []

               split_data["ydata"][yname].append(data["ydata"][0])
               split_data["xdata"][yname].append(data["xdata"][0])

      fname = "{}_{}.png".format(outputfile_name, split_name)
      if gname:
         gname = "{}\n{}".format(split_name, gname)
      else:
         gname = "{}".format(split_name)

      print("Plotting {}:".format(fname))

      for yname in split_data["ynames"]:

         ycolor = split_data["ycolor"][yname]
         yformat = split_data["yformat"][yname]
         ydata = split_data["ydata"][yname]
         xdata = split_data["xdata"][yname]

         if yname in dc.g_series_reformat and "format" in dc.g_series_reformat[yname]:
            yformat = dc.g_series_reformat[yname]["format"]
         if yname in dc.g_series_reformat and "color" in dc.g_series_reformat[yname]:
            ycolor = dc.g_series_reformat[yname]["color"]

         print("  series \"{}\" format \"{}\" color \"{}\"".format(yname, yformat, ycolor))

         if len(ykinds) > 1:
            yname = "{} {}".format(dc.Data.kinds[ykind].kind, yname)
         np_xdata = np.array(xdata)
         xind = np_xdata.argsort()
         np_xdata = np_xdata[xind[0:]]
         np_ydata = np.array(ydata)
         np_ydata = np_ydata[xind[0:]]
         plt.plot(np_xdata,np_ydata,yformat,color=ycolor,label=yname)

      if ylabel:
         plt.ylabel(ylabel)
      if yscale:
         plt.yscale(yscale)
      if ylim:
         plt.ylim(ylim)

      if xlabel:
         plt.xlabel(xlabel)
      if xscale:
         plt.xscale(xscale)
      if xlim:
         plt.xlim(xlim)
      #print(plt.rcParams.keys())
      plt.title(gname)
      handles, labels = plt.gca().get_legend_handles_labels()
      legend_order = dc.set_legend_order(labels)
      plt.legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order],loc=lloc)
      plt.grid(True)

      plt.savefig(fname, dpi=150.0)
      plt.clf()

def plot_data_bar(outputfile_name, xaxis, ykinds):
   # print("plotting {} {} {}".format(outputfile_name, xaxis, ykinds))

   assert(xaxis == "kernel_index")

   gname = g_gname

   lloc = g_lloc

   xlabel = g_xlabel or "Kernel"
   xscale = g_xscale
   xlim = g_xlim

   ylabel = g_ylabel
   yscale = g_yscale
   ylim = g_ylim

   for ykind in ykinds:
      if gname:
         gname = "{}\n{}".format(gname, ykind)
      else:
         gname = "{}".format(ykind)
      if not ykind in dc.Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not ylabel:
         ylabel = dc.Data.kinds[ykind].label
      elif (not g_ylabel) and ylabel != dc.Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([dc.Data.kinds[_ykind].label for _ykind in ykinds]))

   kernel_data = { "kernel_names": [],
                   "kernel_centers": [],
                   "ynames": {},
                   "ycolor": {},
                   "ydata": {}, }

   for kernel_index in range(0, dc.Data.num_kernels):
      kernel_name = dc.Data.kernels[kernel_index]

      kernel_data["kernel_names"].append(kernel_name)
      kernel_data["kernel_centers"].append(kernel_index)

      axes_index = { dc.Data.axes["kernel_index"]: kernel_index }

      for ykind in ykinds:

         ydata_list = get_plot_data(ykind, axes_index)

         for ydata in ydata_list:

            assert(len(ydata["data"]) == 1)

            yname = ydata["name"]
            if len(ykinds) > 1:
               yname = "{} {}".format(dc.Data.kinds[ykind].kind, yname)

            ycolor = (0.0, 0.0, 0.0, 1.0)
            if dc.Data.axes["variant_index"] in ydata["axes_index"]:
               variant_index = ydata["axes_index"][dc.Data.axes["variant_index"]]
               ycolor = dc.Data.variant_colors[variant_index]

            if not yname in kernel_data["ynames"]:
               kernel_data["ynames"][yname] = len(kernel_data["ynames"])
               kernel_data["ycolor"][yname] = ycolor
               kernel_data["ydata"][yname] = []

            # pad with 0s if find missing data
            while len(kernel_data["ydata"][yname])+1 < len(kernel_data["kernel_names"]):
               kernel_data["ydata"][yname].append(0.0)

            kernel_data["ydata"][yname].append(ydata["data"][0])

   fname = "{}.png".format(outputfile_name)
   if not gname:
      gname = "{}".format("bar")

   print("Plotting {}:".format(fname))

   num_xticks = len(kernel_data["kernel_centers"])
   plt.figure(figsize=(max(num_xticks*0.5, 4), 6,))

   y_n = len(kernel_data["ydata"])
   ywidth = 1.0 / (y_n+1)
   for yname in kernel_data["ynames"]:

      y_i = kernel_data["ynames"][yname]
      ycolor = kernel_data["ycolor"][yname]
      yaxis = kernel_data["ydata"][yname]

      if yname in dc.g_series_reformat and "color" in dc.g_series_reformat[yname]:
         ycolor = dc.g_series_reformat[yname]["color"]

      print("  series \"{}\" color \"{}\"".format(yname, ycolor))

      xaxis = [c + (y_i+1)/(y_n+1) - 0.5 for c in kernel_data["kernel_centers"]]

      # pad with 0s if find missing data
      while len(yaxis) < len(kernel_data["kernel_names"]):
         yaxis.append(0.0)

      plt.bar(xaxis,yaxis,label=yname,width=ywidth,color=ycolor,zorder=2) # ,edgecolor="grey")

   xticks = kernel_data["kernel_centers"]
   xtick_names = kernel_data["kernel_names"]

   if ylabel:
      plt.ylabel(ylabel)
   if yscale:
      plt.yscale(yscale)
   if ylim:
      plt.ylim(ylim)

   if xlabel:
      plt.xlabel(xlabel)
   if xscale:
      plt.xscale(xscale)
   if xlim:
      plt.xlim(xlim)

   plt.xticks(xticks, xtick_names, rotation=90)

   plt.title(gname)
   plt.legend(loc=lloc)
   plt.grid(True, zorder=0)

   plt.savefig(fname, dpi=150.0, bbox_inches="tight")
   plt.clf()


def plot_data_histogram(outputfile_name, haxis, hkinds):
   # print("plotting {} {} {}".format(outputfile_name, haxis, hkinds))

   assert(haxis == "kernel_index")

   gname = g_gname

   lloc = g_lloc

   hbin_size = g_hbin_size
   hbin_max = None
   hbin_min = None

   xlabel = g_xlabel
   xscale = g_xscale
   xlim = g_xlim

   ylabel = g_ylabel or "Number"
   yscale = g_yscale
   ylim = g_ylim

   for ykind in hkinds:
      if gname:
         gname = "{}\n{}".format(gname, ykind)
      else:
         gname = "{}".format(ykind)
      if not ykind in dc.Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not xlabel:
         xlabel = dc.Data.kinds[ykind].label
      elif (not g_xlabel) and xlabel != dc.Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([dc.Data.kinds[_ykind].label for _ykind in hkinds]))

   if not hbin_size:

      hdata_all = []

      for kernel_index in range(0, dc.Data.num_kernels):
         kernel_name = dc.Data.kernels[kernel_index]

         axes_index = { dc.Data.axes["kernel_index"]: kernel_index }

         for ykind in hkinds:

            hdata_list = get_plot_data(ykind, axes_index)

            for hdata in hdata_list:

               assert(len(hdata["data"]) == 1)
               hdata_all.append(hdata["data"][0])

      hdata_all.sort()

      num_hdata = len(hdata_all)
      i_Q1 = math.floor(num_hdata * 0.25)
      i_Q3 = math.floor(num_hdata * 0.75)
      hdata_Q1 = hdata_all[i_Q1]
      hdata_Q3 = hdata_all[i_Q3]
      iqr = hdata_Q3 - hdata_Q1

      hbin_size = 2.0 * iqr / num_hdata**(1.0/3.0)

      if hbin_size > 1.0:
         hbin_size = math.floor(hbin_size)
      elif hbin_size > 0.0:
         hbin_size = 1.0 / math.ceil(1.0 / hbin_size)
      else:
         hbin_size = 1.0

   kernel_data = { "hnames": {},
                   "hcolor": {},
                   "hbins": {}, }

   for kernel_index in range(0, dc.Data.num_kernels):
      kernel_name = dc.Data.kernels[kernel_index]

      axes_index = { dc.Data.axes["kernel_index"]: kernel_index }

      for ykind in hkinds:

         hdata_list = get_plot_data(ykind, axes_index)

         for hdata in hdata_list:

            assert(len(hdata["data"]) == 1)

            hname = hdata["name"]
            if len(hkinds) > 1:
               hname = "{} {}".format(dc.Data.kinds[ykind].kind, hname)

            hcolor = (0.0, 0.0, 0.0, 1.0)
            if dc.Data.axes["variant_index"] in hdata["axes_index"]:
               variant_index = hdata["axes_index"][dc.Data.axes["variant_index"]]
               hcolor = dc.Data.variant_colors[variant_index]

            if not hname in kernel_data["hnames"]:
               kernel_data["hnames"][hname] = len(kernel_data["hnames"])
               kernel_data["hcolor"][hname] = hcolor
               kernel_data["hbins"][hname] = {}

            hbin = math.floor(hdata["data"][0] / hbin_size)

            if hbin_max == None or hbin > hbin_max:
               hbin_max = hbin
            if hbin_min == None or hbin < hbin_min:
               hbin_min = hbin

            if not hbin in kernel_data["hbins"][hname]:
               kernel_data["hbins"][hname][hbin] = 0
            kernel_data["hbins"][hname][hbin] += 1

   fname = "{}.png".format(outputfile_name)
   if not gname:
      gname = "{}".format("histogram")

   print("Plotting {}:".format(fname))

   num_xticks = hbin_max - hbin_min + 1
   if xlim:
      num_xticks = math.ceil((xlim[1] - xlim[0]) / hbin_size)
   plt.figure(figsize=(max(num_xticks*0.5, 4), 6,))

   h_n = len(kernel_data["hnames"])
   hwidth = hbin_size / h_n
   #print(h_n, hwidth, hbin_size)
   for hname in kernel_data["hnames"]:

      h_i = kernel_data["hnames"][hname]
      xoffset = hbin_size * ((h_i+0.5)/h_n)
      hcolor = kernel_data["hcolor"][hname]
      hbins = kernel_data["hbins"][hname]

      if hname in dc.g_series_reformat and "color" in dc.g_series_reformat[hname]:
         hcolor = dc.g_series_reformat[hname]["color"]

      print("  series \"{}\" color \"{}\" offset {}".format(hname, hcolor, xoffset))

      xaxis = []
      haxis = []
      for i, hval in hbins.items():
         xval = i * hbin_size + xoffset
         xaxis.append(xval)
         haxis.append(hval)

      plt.bar(xaxis,haxis,label=hname,width=hwidth,color=hcolor,zorder=2) # ,edgecolor="grey")

   if ylabel:
      plt.ylabel(ylabel)
   if yscale:
      plt.yscale(yscale)
   if ylim:
      plt.ylim(ylim)

   if xlabel:
      plt.xlabel(xlabel)
   if xscale:
      plt.xscale(xscale)
   if xlim:
      plt.xlim(xlim)

   plt.title(gname)
   plt.legend(loc=lloc)
   plt.grid(True, zorder=0)

   plt.savefig(fname, dpi=150.0, bbox_inches="tight")
   plt.clf()


def main(argv):
   outputfile = "graph"
   runinfo_filename = dc.g_runinfo_filename
   timing_filename = dc.g_timing_filename
   print_kinds = []
   split_line_graph_kind_lists = []
   bar_graph_kind_lists = []
   histogram_graph_kind_lists = []
   
   
   # set a few plot params - see rcParams.keys() for list
   params = {'xtick.labelsize':'small',
             'ytick.labelsize':'small',
             'axes.labelsize':'small',
             'axes.titlesize':'medium',
             'legend.fontsize':'x-small'}
   plt.rcParams.update(params)
   
  
   parser = ac.process_argparse()
   args, unknown = parser.parse_args(argv)
   print(args)

   cr = None
   # argparse module can do a Hatchet Caliper Reader check for arg --caliper
   # arg.caliper is False by default
   can_process_caliper = args.caliper
   # cr is set to importlib hatchet
   if can_process_caliper:
      cr = args.cr

   #kernels section
   parse_set = set()
   if args.kernels != None:
      parse_set.update(set(args.kernels))
   if args.kernels_close != None:
      parse_set.update(set(args.kernels_close))
   for k in list(parse_set):
      print("including kernel:" + str(k))
      dc.Data.include_kernels[k] = k
      
   parse_set = set()
   if args.exclude_kernels != None:
      parse_set.update(set(args.exclude_kernels))
   if args.exclude_kernels_close != None:
      parse_set.update(set(args.exclude_kernels_close))
   for k in list(parse_set):
      print("excluding kernel:" + str(k))
      dc.Data.exclude_kernels[k] = k

   # variant section
   parse_set = set()
   if args.variants != None:
      parse_set.update(set(args.variants))
   if args.variants_close != None:
      parse_set.update(set(args.variants_close))
   for k in list(parse_set):
      print("including variant:" + str(k))
      dc.Data.include_variants[k] = k

   parse_set = set()
   if args.exclude_variants != None:
      parse_set.update(set(args.exclude_variants))
   if args.exclude_variants_close != None:
      parse_set.update(set(args.exclude_variants_close))
   for k in list(parse_set):
      print("excluding variant:" + str(k))
      dc.Data.exclude_variants[k] = k
   
   #tuning section
   parse_set = set()
   if args.tunings != None:
      parse_set.update(set(args.tunings))
   if args.tunings_close != None:
      parse_set.update(set(args.tunings_close))
   for k in list(parse_set):
      print("including tuning:" + str(k))
      dc.Data.include_tunings[k] = k

   parse_set = set()
   if args.exclude_tunings != None:
      parse_set.update(set(args.exclude_tunings))
   if args.exclude_tunings_close != None:
      parse_set.update(set(args.exclude_tunings_close))
   for k in list(parse_set):
      print("excluding tuning:" + str(k))
      dc.Data.exclude_tunings[k] = k
      
   sweep_dir_paths = args.prescan["directories"]
   
   if args.output != None:
      outputfile = args.output[0]
      
   if args.graph_name != None:
      global g_gname
      g_gname = args.graph_name[0]
      
   if args.legend_location != None:
      global g_lloc
      g_lloc = (float(args.legend_location[0]), float(args.legend_location[1]))
      
   if args.y_axis_label != None:
      global g_ylabel
      g_ylabel = args.y_axis_label[0]
      
   if args.y_axis_scale != None:
      global g_yscale
      g_yscale = args.y_axis_scale[0]
      
   if args.y_axis_limit != None:
      global g_ylim
      g_ylim = (float(args.y_axis_limit[0]),float(args.y_axis_limit[1]))

   if args.x_axis_label != None:
      global g_xlabel
      g_xlabel = args.x_axis_label[0]

   if args.x_axis_scale != None:
      global g_xscale
      g_xscale = args.x_axis_scale[0]

   if args.x_axis_limit != None:
      global g_xlim
      g_xlim = (float(args.x_axis_limit[0]), float(args.x_axis_limit[1]))
      
   if args.recolor != None:
      # expect one or more repeating sequence of series tuple
      for ll in args.recolor:
         series_name = ll[0]
         # expecting tuple string "(r,g,b)" with r,g,b floats in [0-1]
         tuple_str = ll[1]
         if not series_name in dc.g_series_reformat:
            dc.g_series_reformat[series_name] = {}
         dc.g_series_reformat[series_name]['color'] = dc.make_color_tuple_str(tuple_str)
      
   if args.reformat != None:
      for ll in args.reformat:
         series_name = ll[0]
         format_str = ll[1]
         if not series_name in dc.g_series_reformat:
            dc.g_series_reformat[series_name] = {}
         dc.g_series_reformat[series_name]['format'] = format_str
   
   if args.kernel_groups != None:
      for g in args.kernel_groups:
         dc.Data.include_kernel_groups[g] = g
         
   for kernel_group in dc.Data.include_kernel_groups.keys():
      if kernel_group in dc.g_known_kernel_groups:
         print("include kernel group:"+str(kernel_group))
         for kernel_name in dc.g_known_kernel_groups[kernel_group]["kernels"]:
            if kernel_name in args.prescan["kernels_union"]:
               dc.Data.include_kernels[kernel_name] = kernel_name
      else:
         print("Unknown kernel group {}".format(kernel_group))
         sys.exit(2)
         
   if args.exclude_kernel_groups != None:
      for g in args.exclude_kernel_groups:
         dc.Data.exclude_kernel_groups[g] = g
         
   for kernel_group in dc.Data.exclude_kernel_groups.keys():
      if kernel_group in dc.g_known_kernel_groups:
         print("exclude kernel group:"+str(kernel_group))
         for kernel_name in dc.g_known_kernel_groups[kernel_group]["kernels"]:
            if kernel_name in args.prescan["kernels_union"]:
               dc.Data.exclude_kernels[kernel_name] = kernel_name
      else:
         print("Unknown kernel group {}".format(kernel_group))
         sys.exit(2)
   
   compact_flag = True
   if args.print_compact != None:
      for aa in args.print_compact:
         print_kinds.append(aa)

   if args.print_expanded != None:
      compact_flag = False
      for aa in args.print_expanded:
         print_kinds.append(aa)

   if args.split_line_graphs != None:
      split_line_graph_kind_lists.append([])
      for aa in args.split_line_graphs:
         split_line_graph_kind_lists[len(split_line_graph_kind_lists) - 1].append(aa)
         
   if args.bar_graph != None:
      bar_graph_kind_lists.append([])
      for aa in args.bar_graph:
         bar_graph_kind_lists[len(bar_graph_kind_lists) - 1].append(aa)
         
   if args.histogram_graph != None:
      histogram_graph_kind_lists.append([])
      for aa in args.histogram_graph:
         histogram_graph_kind_lists[len(histogram_graph_kind_lists) - 1].append(aa)
         
   #done with options
   print("Input directories are \"{0}\"".format(sweep_dir_paths))
   print("Output file is \"{0}\"".format(outputfile))
   for sweep_dir_path in sweep_dir_paths:
      print("sweep_dir_path:" + sweep_dir_path)
      sweep_dir_name = os.path.basename(sweep_dir_path)
      print("sweep_dir_name:" + sweep_dir_name)

      if sweep_dir_name in dc.Data.exclude_sweeps:
         continue

      if sweep_dir_name in dc.Data.sweeps:
         raise NameError("Repeated sweep_dir_name {}".format(sweep_dir_name))
      dc.Data.add_sweep(sweep_dir_name)
      sweep_index = dc.Data.sweeps[sweep_dir_name]
      if sweep_index >= len(dc.g_markers):
         raise NameError("Ran out of sweep markers for {}".format(sweep_dir_name))
      dc.Data.sweep_markers[sweep_index] = dc.g_markers[sweep_index]

      for r0,sweep_subdir_names,f0 in os.walk(sweep_dir_path):
         for sweep_subdir_name in sweep_subdir_names:
            sweep_subdir_path = os.path.join(sweep_dir_path, sweep_subdir_name)
            # print(sweep_dir_name, sweep_subdir_path)

            run_size_name = ac.get_size_from_dir_name(sweep_subdir_name)
            if run_size_name in args.prescan["sweep_sizes"]:
               if not run_size_name in dc.Data.run_sizes:
                  dc.Data.add_run_size(run_size_name)
               run_size_index = dc.Data.run_sizes[run_size_name]
            else:
               continue

            sweep_subdir_timing_file_path = ""
            sweep_subdir_runinfo_file_path = ""
            for r1,d1,sweep_subdir_file_names in os.walk(sweep_subdir_path):
               for sweep_subdir_file_name in sweep_subdir_file_names:
                  sweep_subdir_file_path = os.path.join(sweep_subdir_path, sweep_subdir_file_name)
                  if sweep_subdir_file_name == timing_filename:
                     sweep_subdir_timing_file_path = sweep_subdir_file_path
                  elif sweep_subdir_file_name == runinfo_filename:
                     sweep_subdir_runinfo_file_path = sweep_subdir_file_path

            if sweep_subdir_timing_file_path != "" and sweep_subdir_runinfo_file_path != "":
               #print(sweep_subdir_timing_file_path, sweep_subdir_runinfo_file_path)
               #read_runinfo_file(sweep_index, sweep_subdir_runinfo_file_path, run_size_index)
               if(can_process_caliper):
                  read_caliper_runinfo_file(cr,sweep_index, sweep_subdir_path, run_size_index)
                  read_caliper_timing_file(cr,sweep_index, sweep_subdir_path, run_size_index)
               else:
                  read_runinfo_file(sweep_index, sweep_subdir_runinfo_file_path, run_size_index)
                  read_timing_file(sweep_index, sweep_subdir_timing_file_path, run_size_index)


   kinds_string = ""
   for kindTree in dc.Data.kinds.values():
      kinds_string += ", {}".format(kindTree.kind)
   print("kinds")
   print("  {}".format(kinds_string[2:]))

   kind_templates_string = ""
   for kindTree_template in dc.Data.kind_templates.values():
      kind_templates_string += ", {}".format(kindTree_template.kind_template)
   print("kind_templates")
   print("  {}".format(kind_templates_string[2:]))

   axes_string = ""
   for v in range(0, dc.Data.num_axes):
      axes_string += ", {}".format(dc.Data.axes[v])
   print("axes")
   print("  {}".format(axes_string[2:]))

   sweeps_string = ""
   for v in range(0, dc.Data.num_sweeps):
      sweeps_string += ", {}".format(dc.Data.sweeps[v])
   print("sweeps")
   print("  {}".format(sweeps_string[2:]))

   run_sizes_string = ""
   for v in range(0, dc.Data.num_run_sizes):
      run_sizes_string += ", {}".format(dc.Data.run_sizes[v])
   print("run_sizes")
   print("  {}".format(run_sizes_string[2:]))

   kernel_groups_string = ""
   for kernel_group in dc.g_known_kernel_groups:
      kernel_groups_string += ", {}".format(kernel_group)
   print("kernel groups")
   print("  {}".format(kernel_groups_string[2:]))

   assert dc.Data.num_kernels > 0,f"Expected kernels to be greater than zero; kernel name typo in cmdline arg??"
   kernel_string = ""
   for v in range(0, dc.Data.num_kernels):
      kernel_string += ", {}".format(dc.Data.kernels[v])
   print("kernels")
   print("  {}".format(kernel_string[2:]))

   variant_string = ""
   for v in range(0, dc.Data.num_variants):
      variant_string += ", {}".format(dc.Data.variants[v])
   print("variants")
   print("  {}".format(variant_string[2:]))

   tuning_string = ""
   for v in range(0, dc.Data.num_tunings):
      tuning_string += ", {}".format(dc.Data.tunings[v])
   print("tunings")
   print("  {}".format(tuning_string[2:]))

   for kind in print_kinds:
      print("Print Data {}:".format(kind))
      dc.Data.compute(kind)
      print(dc.Data.kinds[kind].dataString(compact_flag))

   for kind_list in split_line_graph_kind_lists:
      print("Plot split line graph {}:".format(kind_list))
      for kind in kind_list:
         dc.Data.compute(kind)
      plot_data_split_line(outputfile, "kernel_index", "run_size", "Problem size", kind_list)

   for kind_list in bar_graph_kind_lists:
      print("Plot bar graph {}:".format(kind_list))
      for kind in kind_list:
         dc.Data.compute(kind)
      plot_data_bar(outputfile, "kernel_index", kind_list)

   for kind_list in histogram_graph_kind_lists:
      print("Plot histogram graph {}:".format(kind_list))
      for kind in kind_list:
         dc.Data.compute(kind)
      plot_data_histogram(outputfile, "kernel_index", kind_list)

if __name__ == "__main__":
   main(sys.argv[1:])
