#!/usr/bin/env python3

import math
import os
import sys
import getopt
import csv
import matplotlib.pyplot as plt

def normalize_color_tuple(t):
   len_t = 0.0
   for i in range(0, len(t)):
      len_t += t[i]*t[i]
   len_t = math.sqrt(len_t)
   new_t = ()
   for i in range(0, len(t)):
      new_t += (t[i]/len_t,)
   return new_t

def clamp_tuple(t, min_val=0.0, max_val=1.0):
   new_t = ()
   for i in range(0, len(t)):
      val = t[i]
      if val > max_val:
         val = max_val
      elif val < min_val:
         val = min_val
      new_t += (val,)
   return new_t

def color_mul(t, factor):
   new_t = ()
   for i in range(0, len(t)):
      new_t += (t[i]*factor,)
   return clamp_tuple(new_t)

def make_color_tuple(r, g, b):
   return (r/255.0, g/255.0, b/255.0)

g_color_base_factor = 1.0
g_color_lambda_factor = 0.7
g_color_raja_factor = 0.4
g_color_seq = normalize_color_tuple(make_color_tuple(204, 119, 34)) # ocre
g_color_omp = normalize_color_tuple(make_color_tuple(0, 115, 125)) # omp teal
g_color_ompt = normalize_color_tuple(make_color_tuple(125, 10, 0)) # omp teal compliment
g_color_cuda = normalize_color_tuple(make_color_tuple(118, 185, 0)) # nvidia green
g_color_hip = normalize_color_tuple(make_color_tuple(237, 28, 36)) # amd red
g_known_variants = { "Base_Seq": {"color": color_mul(g_color_seq, g_color_base_factor)},
                     "Lambda_Seq": {"color": color_mul(g_color_seq, g_color_lambda_factor)},
                     "RAJA_Seq": {"color": color_mul(g_color_seq, g_color_raja_factor)},

                     "Base_OpenMP": {"color": color_mul(g_color_omp, g_color_base_factor)},
                     "Lambda_OpenMP": {"color": color_mul(g_color_omp, g_color_lambda_factor)},
                     "RAJA_OpenMP": {"color": color_mul(g_color_omp, g_color_raja_factor)},

                     "Base_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_base_factor)},
                     "Lambda_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_lambda_factor)},
                     "RAJA_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_raja_factor)},

                     "Base_CUDA": {"color": color_mul(g_color_cuda, g_color_base_factor)},
                     "Lambda_CUDA": {"color": color_mul(g_color_cuda, g_color_lambda_factor)},
                     "RAJA_CUDA": {"color": color_mul(g_color_cuda, g_color_raja_factor)},

                     "Base_HIP": {"color": color_mul(g_color_hip, g_color_base_factor)},
                     "Lambda_HIP": {"color": color_mul(g_color_hip, g_color_lambda_factor)},
                     "RAJA_HIP": {"color": color_mul(g_color_hip, g_color_raja_factor)}
                   }
g_known_tunings =  { "default": {"format": "-"},
                     "block_25": {"format": "-"},
                     "block_32": {"format": ":"},
                     "block_64": {"format": "-."},
                     "block_128": {"format": "--"},
                     "block_256": {"format": "-"},
                     "block_512": {"format": "-."},
                     "block_1024": {"format": "-"},
                     "cub": {"format": ":"},
                     "rocprim": {"format": ":"}
                   }
g_markers = [ "o", "s", "+", "x", "*", "d", "h", "p", "8" ]

g_timing_filename = "RAJAPerf-timing-Minimum.csv"
g_runinfo_filename = "RAJAPerf-kernels.csv"
g_timing_file_kind = "time(s)"

# multi-dimensional array structured like this
#   kind - time, bandwidth, etc (["info"])
#     directory name - platform, compiler, etc
#       run size - problem size, for g_run_sizes
#         kernel index - for g_kernels
g_info = { "Problem size": { },
           "Reps": { },
           "Iterations/rep": { },
           "Kernels/rep": { },
           "Bytes/rep": { },
           "FLOPS/rep": { }
         }

# multi-dimensional array structured like this
#   kind - time, bandwidth, etc (["data"])
#     directory name - platform, compiler, etc
#       run size - problem size, for g_run_sizes
#         kernel index - for g_kernels
#           variant index - for g_variants
#             tuning index - for g_tunings
g_data = { }

g_model_kind = "time(s)"

# has info derivable from first kind "time(s)" which is read from files
g_kinds = { "Problem size": { "type": "info" },
            "Reps": { "type": "info" },
            "Iterations/rep": { "type": "info" },
            "Kernels/rep": { "type": "info" },
            "Bytes/rep": { "type": "info" },
            "FLOPS/rep": { "type": "info" },

            "time(s)" : { "type": "data" },

            "time(ms)": { "type": "computed",
                          "args": ["time(s)"],
                          "func": lambda t: t * 1000.0 },
            "time(us)": { "type": "computed",
                          "args": ["time(s)"],
                          "func": lambda t: t * 1000000.0 },
            "time(ns)": { "type": "computed",
                          "args": ["time(s)"],
                          "func": lambda t: t * 1000000000.0 },

            "time/rep(s)":  { "type": "computed",
                              "args": ["time(s)", "Reps"],
                              "func": lambda t, r: t / r },
            "time/rep(ms)": { "type": "computed",
                              "args": ["time/rep(s)"],
                              "func": lambda tpr: tpr * 1000.0 },
            "time/rep(us)": { "type": "computed",
                              "args": ["time/rep(s)"],
                              "func": lambda tpr: tpr * 1000000.0 },
            "time/rep(ns)": { "type": "computed",
                              "args": ["time/rep(s)"],
                              "func": lambda tpr: tpr * 1000000000.0 },

            "time/it(s)":  { "type": "computed",
                             "args": ["time/rep(s)", "Iterations/rep"],
                             "func": lambda tpr, ipr: tpr / ipr },
            "time/it(ms)": { "type": "computed",
                             "args": ["time/it(s)"],
                             "func": lambda tpi: tpi * 1000.0 },
            "time/it(us)": { "type": "computed",
                             "args": ["time/it(s)"],
                             "func": lambda tpi: tpi * 1000000.0 },
            "time/it(ns)": { "type": "computed",
                             "args": ["time/it(s)"],
                             "func": lambda tpi: tpi * 1000000000.0 },

            "time/kernel(s)":  { "type": "computed",
                                 "args": ["time/rep(s)", "Kernels/rep"],
                                 "func": lambda tpr, kpr: tpr / kpr },
            "time/kernel(ms)": { "type": "computed",
                                 "args": ["time/kernel(s)"],
                                 "func": lambda tpi: tpi * 1000.0 },
            "time/kernel(us)": { "type": "computed",
                                 "args": ["time/kernel(s)"],
                                 "func": lambda tpi: tpi * 1000000.0 },
            "time/kernel(ns)": { "type": "computed",
                                 "args": ["time/kernel(s)"],
                                 "func": lambda tpi: tpi * 1000000000.0 },

            "throughput(Problem size/s)":  { "type": "computed",
                                             "args": ["time/rep(s)", "Problem size"],
                                             "func": lambda tpr, ps: ps / tpr },
            "throughput(KProblem size/s)": { "type": "computed",
                                             "args": ["throughput(Problem size/s)"],
                                             "func": lambda thr: thr / 1000.0 },
            "throughput(MProblem size/s)": { "type": "computed",
                                             "args": ["throughput(Problem size/s)"],
                                             "func": lambda thr: thr / 1000000.0 },
            "throughput(GProblem size/s)": { "type": "computed",
                                             "args": ["throughput(Problem size/s)"],
                                             "func": lambda thr: thr / 1000000000.0 },
            "throughput(TProblem size/s)": { "type": "computed",
                                             "args": ["throughput(Problem size/s)"],
                                             "func": lambda thr: thr / 1000000000000.0 },

            "bandwidth(B/s)":   { "type": "computed",
                                  "args": ["time/rep(s)", "Bytes/rep"],
                                  "func": lambda tpr, bpr: bpr / tpr },
            "bandwidth(Kb/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1000.0 },
            "bandwidth(MB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1000000.0 },
            "bandwidth(GB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1000000000.0 },
            "bandwidth(TB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1000000000000.0 },

            "bandwidth(B/s)":   { "type": "computed",
                                  "args": ["time/rep(s)", "Bytes/rep"],
                                  "func": lambda tpr, bpr: bpr / tpr },
            "bandwidth(Kib/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1024.0 },
            "bandwidth(MiB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1048576.0 },
            "bandwidth(GiB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1073741824.0 },
            "bandwidth(TiB/s)": { "type": "computed",
                                  "args": ["bandwidth(B/s)"],
                                  "func": lambda bps: bps / 1099511627776.0 },

            "FLOPS":  { "type": "computed",
                        "args": ["time/rep(s)", "FLOPS/rep"],
                        "func": lambda tpr, fpr: fpr / tpr },
            "MFLOPS": { "type": "computed",
                        "args": ["FLOPS"],
                        "func": lambda fps: fps / 1000.0 },
            "GFLOPS": { "type": "computed",
                        "args": ["FLOPS"],
                        "func": lambda fps: fps / 1000000.0 },
            "TFLOPS": { "type": "computed",
                        "args": ["FLOPS"],
                        "func": lambda fps: fps / 1000000000.0 },
         }

g_num_sweeps = 0
g_sweeps = {}
g_sweep_markers = {}

g_num_run_sizes = 0
g_run_sizes = {}

g_num_kernels = 0
g_kernels = {}

g_num_variants = 0
g_variants = {}
g_variant_colors = {}

g_num_tunings = 0
g_tunings = {}
g_tuning_formats = {}

def get_size_from_dir_name(sweep_subdir_name):
   # print(sweep_subdir_name)
   return int(sweep_subdir_name.replace("SIZE_", ""));

def read_runinfo_file(sweep_dir_name, sweep_subdir_runinfo_file_path, run_size):
   # print(sweep_dir_name, sweep_subdir_runinfo_file_path, run_size)
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
               # print(c, info_kind)
               if not info_kind in g_kinds:
                  print("Unknown kernel info {0}".format(info_kind))
                  g_kinds[info_kind] = { "type": "info" }
               if info_kind in c_to_info_kinds:
                  print("Repeated kernel info {0}".format(info_kind))
                  sys.exit(1)
               if not info_kind in g_info:
                  # add new kind to global data
                  g_info[info_kind] = {}
               if not sweep_dir_name in g_info[info_kind]:
                  # add new sweep to global data
                  g_info[info_kind][sweep_dir_name] = {}
               if run_size in g_info[info_kind][sweep_dir_name]:
                  print("Repeated kernel size {0} in {1}".format(sweep_dir_name, run_size))
                  sys.exit(1)
               else:
                  # add new size to global data
                  g_info[info_kind][sweep_dir_name][run_size] = {}
               # make map of columns to names
               c_to_info_kinds[c] = info_kind
               c_to_info_kinds[info_kind] = c
         elif not ignore:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in g_kernels:
               kernel_index = g_kernels[kernel_name]
            else:
               # add kernel to global list
               global g_num_kernels
               kernel_index = g_num_kernels
               g_num_kernels += 1
               g_kernels[kernel_name]  = kernel_index
               g_kernels[kernel_index] = kernel_name

            for c in range(1, len(row)):
               info_kind = c_to_info_kinds[c]
               try:
                  # add data to global structure
                  val = int(row[c].strip())
                  # print(kernel_index, kernel_name, info_kind, val)
                  g_info[info_kind][sweep_dir_name][run_size][kernel_index] = val
               except ValueError:
                  pass # could not convert data to int

def read_timing_file(sweep_dir_name, sweep_subdir_timing_file_path, run_size):
   # print(sweep_dir_name, sweep_subdir_timing_file_path, run_size)
   with open(sweep_subdir_timing_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      data_kind = g_timing_file_kind
      if not data_kind in g_data:
         g_data[data_kind] = {}
      if not sweep_dir_name in g_data[data_kind]:
         g_data[data_kind][sweep_dir_name] = {}
      if not run_size in g_data[data_kind][sweep_dir_name]:
         g_data[data_kind][sweep_dir_name][run_size] = {}
      else:
         raise NameError("Already seen {0} in {1}".format(sweep_dir_name, run_size))

      c_to_variant_index = {}
      c_to_tuning_index = {}
      for row in file_reader:
         # print(row)
         if row[0].strip() == "Kernel":
            if len(c_to_variant_index) == 0:
               for c in range(1, len(row)):
                  variant_name = row[c].strip()
                  if variant_name in g_variants:
                     pass
                  else:
                     global g_num_variants
                     variant_index = g_num_variants
                     g_num_variants += 1
                     g_variants[variant_name]  = variant_index
                     g_variants[variant_index] = variant_name
                     if variant_name in g_known_variants:
                        variant_color = g_known_variants[variant_name]["color"]
                        g_variant_colors[variant_name] = variant_color
                        g_variant_colors[variant_index] = variant_color
                     else:
                        print("Unknown variant {0}".format(variant_name))
                        sys.exit(1)
                  c_to_variant_index[c] = g_variants[variant_name]
            elif len(c_to_tuning_index) == 0:
               for c in range(1, len(row)):
                  tuning_name = row[c].strip()
                  if tuning_name in g_tunings:
                     pass
                  else:
                     global g_num_tunings
                     tuning_index = g_num_tunings
                     g_num_tunings += 1
                     g_tunings[tuning_name]  = tuning_index
                     g_tunings[tuning_index] = tuning_name
                     if tuning_name in g_known_tunings:
                        tuning_format = g_known_tunings[tuning_name]["format"]
                        g_tuning_formats[tuning_name] = tuning_format
                        g_tuning_formats[tuning_index] = tuning_format
                     else:
                        print("Unknown tuning {0}".format(tuning_name))
                        sys.exit(1)
                  c_to_tuning_index[c] = g_tunings[tuning_name]
            else:
               print("Unknown row {0}".format(row))
               sys.exit(1);
         elif len(c_to_variant_index) > 0 and len(c_to_tuning_index) > 0:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in g_kernels:
               kernel_index = g_kernels[kernel_name]
            else:
               raise NameError("Unknown Kernel {0}".format(kernel_name))

            g_data[data_kind][sweep_dir_name][run_size][kernel_index] = {}
            for c in range(1, len(row)):
               variant_index = c_to_variant_index[c]
               tuning_index = c_to_tuning_index[c]
               try:
                  val = float(row[c].strip())
                  # print(kernel_index, kernel_name, variant_index, tuning_index, data_kind, val)
                  if not variant_index in g_data[data_kind][sweep_dir_name][run_size][kernel_index]:
                     g_data[data_kind][sweep_dir_name][run_size][kernel_index][variant_index] = {}
                  g_data[data_kind][sweep_dir_name][run_size][kernel_index][variant_index][tuning_index] = val
               except ValueError:
                  pass # could not convert data to float

def compute_data(kind):
   if not kind in g_kinds:
      raise NameError("Unknown data kind {0}".format(kind))

   if g_kinds[kind]["type"] == "info":
      if kind in g_info:
         return # already calculated
      else:
         raise NameError("Invalid info kind {0}".format(kind))
   elif g_kinds[kind]["type"] == "data":
      if kind in g_data:
         return # already calculated

   if not g_kinds[kind]["type"] == "computed":
      raise NameError("Invalid data kind {0}".format(kind))

   if not ("func" in g_kinds[kind] and "args" in g_kinds[kind]):
      raise NameError("Computing data is not yet supported for kind {0}".format(kind))

   compute_args = g_kinds[kind]["args"]
   compute_func = g_kinds[kind]["func"]

   for arg_kind in compute_args:
      # calculate data for arg_kind
      compute_data(arg_kind)

   model_kind = g_model_kind
   compute_data(model_kind)
   if not model_kind in g_data:
      raise NameError("Model data not available {0}, no args".format(model_kind))

   g_data[kind] = {}
   for sweep_dir_name, model_sweep_data in g_data[model_kind].items():
      g_data[kind][sweep_dir_name] = {}
      for run_size, model_run_data in model_sweep_data.items():
         g_data[kind][sweep_dir_name][run_size] = {}
         for kernel_index, model_kernel_data in model_run_data.items():
            kernel_name = g_kernels[kernel_index]
            g_data[kind][sweep_dir_name][run_size][kernel_index] = {}
            for variant_index, model_variant_data in model_kernel_data.items():
               variant_name = g_variants[variant_index]
               g_data[kind][sweep_dir_name][run_size][kernel_index][variant_index] = {}
               for tuning_index, model_val in model_variant_data.items():
                  tuning_name = g_tunings[tuning_index]

                  args_val = ()
                  for arg_kind in compute_args:
                     if g_kinds[arg_kind]["type"] == "info":
                        arg_val = g_info[arg_kind][sweep_dir_name][run_size][kernel_index]
                     elif g_kinds[arg_kind]["type"] == "data" or g_kinds[arg_kind]["type"] == "computed":
                        arg_val = g_data[arg_kind][sweep_dir_name][run_size][kernel_index][variant_index][tuning_index]
                     else:
                        raise NameError("Invalid data kind {0}".format(arg_kind))
                     args_val = args_val + (arg_val,)

                  val = compute_func(*args_val)
                  g_data[kind][sweep_dir_name][run_size][kernel_index][variant_index][tuning_index] = val


def get_run_size_data(kind, kernel):

   if not kind in g_kinds:
      raise NameError("Unknown kind {}".format(kind))

   kernel_index = kernel
   if isinstance(kernel, str):
      if kernel in g_kernels:
         kernel_index = g_kernels[kernel]
      else:
         raise NameError("Unknown kernel {}".format(kernel))
   elif isinstance(kernel, int):
      kernel_index = kernel
   else:
      raise NameError("Unknown kernel {}".format(kernel))

   if not kernel_index in g_kernels:
      raise NameError("Unknown kernel {}".format(kernel_index))
   kernel_name = g_kernels[kernel_index]

   data = {}

   kind_meta = g_kinds[kind]
   if kind_meta["type"] == "info":

      if not kind in g_info:
         raise NameError("Unknown info kind {}".format(kind))
      kind_info = g_info[kind]

      for sweep_dir_name, sweep_info in kind_info.items():

         data[sweep_dir_name] = {}
         data[sweep_dir_name][kind] = {"type": "info",
                                       "data": [] }

         for run_size, run_info in sweep_info.items():

            if not kernel_index in run_info:
               raise NameError("Unknown info kernel_index {}".format(kernel_index))

            val = run_info[kernel_index]
            data[sweep_dir_name][kind]["data"].append(val)

   elif kind_meta["type"] == "data" or kind_meta["type"] == "computed":

      if kind_meta["type"] == "computed":
         compute_data(kind)

      if not kind in g_data:
         raise NameError("Unknown data kind {}".format(kind))
      kind_data = g_data[kind]

      for sweep_dir_name, sweep_data in kind_data.items():

         data[sweep_dir_name] = {}

         for run_size, run_data in sweep_data.items():

            if not kernel_index in run_data:
               raise NameError("Unknown info kernel_index {}".format(kernel_index))

            kernel_data = run_data[kernel_index]

            for variant_index, variant_data in kernel_data.items():
               variant_name = g_variants[variant_index]
               for tuning_index, val in variant_data.items():
                  tuning_name = g_tunings[tuning_index]

                  data_name = "{}-{}".format(variant_name, tuning_name)

                  if not data_name in data[sweep_dir_name]:
                     data[sweep_dir_name][data_name] = {"type": "data",
                                                        "variant": variant_index,
                                                        "tuning": tuning_index,
                                                        "data": [] }

                  data[sweep_dir_name][data_name]["data"].append(val)

   else:
      raise NameError("Unknown kind {} type {}".format(kind, kind_meta["type"]))

   return data

def plot_data(outputfile_name, ykind):
   print(outputfile_name, ykind)

   if not ykind in g_kinds:
      raise NameError("Unknown kind {}".format(ykind))

   ylabel = ykind
   yscale = "log"
   ylim = None

   xkind = "Problem size"
   xlabel = xkind
   xscale = "log"
   xlim = None

   for kernel_index in range(0, g_num_kernels):
      kernel_name = g_kernels[kernel_index]

      xaxes = get_run_size_data(xkind, kernel_index)
      yaxes = get_run_size_data(ykind, kernel_index)

      for sweep_index in range(0, g_num_sweeps):
         sweep_dir_name = g_sweeps[sweep_index]
         sweep_marker = g_sweep_markers[sweep_index]

         if not sweep_dir_name in xaxes:
            raise NameError("Unknown sweep_dir_name {}".format(sweep_dir_name))

         for data_name, ydata in yaxes[sweep_dir_name].items():

            yname = "{} {}".format(data_name, sweep_dir_name)
            yformat = "{}-".format(sweep_marker)
            ycolor = (0.0, 0.0, 0.0, 1.0)
            yaxis = ydata["data"]

            if ydata["type"] == "data":
               variant_index = ydata["variant"]
               tuning_index = ydata["tuning"]
               ycolor = g_variant_colors[variant_index]
               yformat = "{}{}".format(sweep_marker, g_tuning_formats[tuning_index])

            if data_name in xaxes[sweep_dir_name]:
               xaxis = xaxes[sweep_dir_name][data_name]["data"]
            elif xkind in xaxes[sweep_dir_name]:
               xaxis = xaxes[sweep_dir_name][xkind]["data"]
            else:
               raise NameError("Unknown xaxis for {}".format(data_name))

            plt.plot(xaxis,yaxis,yformat,color=ycolor,label=yname)

      fname = "{}_{}".format(outputfile_name, kernel_name)
      gname = "{}".format(kernel_name)

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
      plt.legend()
      plt.grid(True)

      plt.savefig(fname, dpi=150.0)
      plt.clf()

def print_data(kind):
   print(kind)

   if kind in g_info:
      kind_info = g_info[kind]
      for sweep_dir_name, sweep_info in kind_info.items():
         print(sweep_dir_name)
         for run_size, run_info in sweep_info.items():
            print(run_size)
            for kernel_index, val in run_info.items():
               kernel_name = g_kernels[kernel_index]
               print(kernel_name, val)

   if kind in g_data:
      kind_data = g_data[kind]
      for sweep_dir_name, sweep_data in kind_data.items():
         print(sweep_dir_name)
         for run_size, run_data in sweep_data.items():
            print(run_size)
            for kernel_index, kernel_data in run_data.items():
               kernel_name = g_kernels[kernel_index]
               for variant_index, variant_data in kernel_data.items():
                  variant_name = g_variants[variant_index]
                  for tuning_index, val in variant_data.items():
                     tuning_name = g_tunings[tuning_index]
                     print(kernel_name, variant_name, tuning_name, val)

def main(argv):
   sweep_dir_paths = []
   outputfile = "output.png"
   help_string = "sweep_graph.py -o <outputfile> <sweepdir1 [sweepdir2 ...]>"
   runinfo_filename = g_runinfo_filename
   timing_filename = g_timing_filename
   print_kinds = []
   graph_kinds = []
   try:
      opts, args = getopt.getopt(argv,"h:o:p:g:",["help","output=","print=","graph="])
   except getopt.GetoptError:
      print(help_string)
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print(help_string)
         sys.exit()
      elif opt in ("-o", "--output"):
         outputfile = arg
      elif opt in ("-p", "--print"):
         print_kinds.append(arg)
      elif opt in ("-g", "--graph"):
         graph_kinds.append(arg)
   for arg in args:
      sweep_dir_paths.append(arg)

   print("Input directories are \"{0}\"".format(sweep_dir_paths))
   print("Output file is \"{0}\"".format(outputfile))

   for sweep_dir_path in sweep_dir_paths:
      sweep_dir_name = os.path.basename(sweep_dir_path)
      # print(sweep_dir_name, sweep_dir_path)

      got_something = False

      for r0,sweep_subdir_names,f0 in os.walk(sweep_dir_path):
         for sweep_subdir_name in sweep_subdir_names:
            sweep_subdir_path = os.path.join(sweep_dir_path, sweep_subdir_name)
            # print(sweep_dir_name, sweep_subdir_path)

            try:
               run_size = get_size_from_dir_name(sweep_subdir_name)

               if not run_size in g_run_sizes:
                  global g_num_run_sizes
                  g_run_sizes[g_num_run_sizes] = run_size
                  g_num_run_sizes += 1

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
                  # print(sweep_subdir_timing_file_path, sweep_subdir_runinfo_file_path)
                  read_runinfo_file(sweep_dir_name, sweep_subdir_runinfo_file_path, run_size)
                  read_timing_file(sweep_dir_name, sweep_subdir_timing_file_path, run_size)
                  got_something = True
            except ValueError:
               print("Couldn't get run size from \"{0}\"".format(sweep_subdir_name))
               pass # could not convert data to int

      if got_something:
         global g_num_sweeps
         if sweep_dir_name in g_sweeps:
            raise NameError("Repeated sweep_dir_name {}".format(sweep_dir_name))
         g_sweeps[g_num_sweeps] = sweep_dir_name
         if g_num_sweeps >= len(g_markers):
            raise NameError("Ran out of sweep markers for {}".format(sweep_dir_name))
         g_sweep_markers[g_num_sweeps] = g_markers[g_num_sweeps]
         g_num_sweeps += 1

   kinds_string = ""
   for kind in g_kinds:
      kinds_string += ", {}".format(kind)
   print("kinds")
   print("  {}".format(kinds_string[2:]))

   run_sizes_string = ""
   for v in range(0, g_num_run_sizes):
      run_sizes_string += ", {}".format(g_run_sizes[v])
   print("run_sizes")
   print("  {}".format(run_sizes_string[2:]))

   kernel_string = ""
   for v in range(0, g_num_kernels):
      kernel_string += ", {}".format(g_kernels[v])
   print("kernels")
   print("  {}".format(kernel_string[2:]))

   variant_string = ""
   for v in range(0, g_num_variants):
      variant_string += ", {}".format(g_variants[v])
   print("variants")
   print("  {}".format(variant_string[2:]))

   tuning_string = ""
   for v in range(0, g_num_tunings):
      tuning_string += ", {}".format(g_tunings[v])
   print("tunings")
   print("  {}".format(tuning_string[2:]))

   for kind in print_kinds:
      print_data(kind)

   for kind in graph_kinds:
      plot_data(outputfile, kind)

if __name__ == "__main__":
   main(sys.argv[1:])
