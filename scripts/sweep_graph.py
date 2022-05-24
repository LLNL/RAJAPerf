#!/usr/bin/env python3

import math
import os
import sys
import re
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


def first(vals):
   return vals[0]

def last(vals):
   return vals[len(vals)-1]

def sum(vals):
   sum_val = 0
   for val in vals:
      sum_val += val
   return sum_val

def avg(vals):
   return sum(vals) / len(vals)

def stddev(vals):
   avg_val = avg(vals)
   stddev_val = 0
   for val in vals:
      stddev_val += (val - avg_val)*(val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val

def relstddev(vals):
   avg_val = avg(vals)
   stddev_val = 0
   for val in vals:
      stddev_val += (val - avg_val)*(val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val / abs(avg_val)

# returns (intercept, slope, correlation_coefficient)
def linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum):
   assert(n>0)
   if n == 1:
      slope = 0.0
      intercept = ysum
      correlation_coefficient = 1.0
   else:
      slope = (n*xysum - xsum*ysum) / (n*x2sum - xsum*xsum)
      intercept = (ysum - slope*xsum)/n
      correlation_coefficient = (n*xysum - xsum*ysum) / math.sqrt((n*x2sum - xsum*xsum)*(n*y2sum - ysum*ysum))
   return (intercept, slope, correlation_coefficient)

# returns (intercept, slope, correlation_coefficient)
def linearRegression(xvals, yvals):
   assert(len(xvals) == len(yvals))
   n = len(xvals)
   xsum = sum(xvals)
   ysum = sum(yvals)
   x2sum = sum([x*x for x in xvals])
   y2sum = sum([y*y for y in yvals])
   xysum = sum([xvals[i]*yvals[i] for i in range(0, n)])
   return linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum)

def eval_linearRegression(xval, lr_vals):
   return lr_vals[0] + lr_vals[1]*xval

# returns (intercept, slope, correlation_coefficient)
def linearRegression_loglog(xvals, yvals):
   assert(len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return linearRegression(xlogvals, ylogvals)

def eval_linearRegression_loglog(xval, lr_vals):
   return math.pow(2, lr_vals[0])*math.pow(xval, lr_vals[1])


# returns (intercept, slope, correlation_coefficient)
def segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR):
   sums[0] += xvals[i]
   sums[1] += yvals[i]
   sums[2] += xvals[i]*xvals[i]
   sums[3] += yvals[i]*yvals[i]
   sums[4] += xvals[i]*yvals[i]
   xsum = sums[0]
   ysum = sums[1]
   x2sum = sums[2]
   y2sum = sums[3]
   xysum = sums[4]
   LR[i] = linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum)

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression_helper(ret, i, n, xvals, yvals, denom, LR_left, LR_right):
   lr_vals_left = None
   lr_vals_right = None
   break_point = math.inf

   if i > 0:
      lr_vals_left = LR_left[i-1]
   if i < n:
      lr_vals_right = LR_right[i]
      break_point = xvals[i]

   numer = 0.0
   for j in range(0, n):
      xval = xvals[j]
      yval = yvals[j]
      lr_vals = None
      if xval < break_point:
         lr_vals = lr_vals_left
      else:
         lr_vals = lr_vals_right
      lr_yval = eval_linearRegression(xval, lr_vals)
      numer += (yval - lr_yval)*(yval - lr_yval)

   correlation_coefficient = 1.0 - numer / denom
   if correlation_coefficient > ret[2]:
      ret[0] = [break_point,]
      ret[1] = [lr_vals_left, lr_vals_right,]
      ret[2] = correlation_coefficient

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression(xvals, yvals):
   assert(len(xvals) == len(yvals))
   N = len(xvals)

   LR_left = []
   LR_right = []
   for i in range(0, N):
      LR_left.append(None)
      LR_right.append(None)

   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for ii in range(0, N):
      i = N-ii-1
      n = ii+1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_right)

   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for i in range(0, N):
      n = i+1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_left)

   yavg = avg(yvals)
   denom = sum([(y-yavg)*(y-yavg) for y in yvals])
   ret = [[], [], -math.inf]
   for i in range(0, N+1):
      segmented_linearRegression_helper(ret, i, N, xvals, yvals, denom, LR_left, LR_right)

   return (*ret,)

def find_segment(xval, break_points):
   break_i = len(break_points)
   for i in range(0, len(break_points)):
      break_point = break_points[i]
      if xval < break_point:
         break_i = i
         break
   return break_i

def eval_segmented_linearRegression(xval, slr_vals):
   break_i = find_segment(xval, slr_vals[0])
   return eval_linearRegression(xval, slr_vals[1][break_i])

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression_loglog(xvals, yvals):
   assert(len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return segmented_linearRegression(xlogvals, ylogvals)

def eval_segmented_linearRegression_loglog(xval, slr_vals):
   break_i = find_segment(math.log(xval, 2), slr_vals[0])
   return eval_linearRegression_loglog(xval, slr_vals[1][break_i])


class Data:

   num_sweeps = 0
   sweeps = {}
   sweep_markers = {}

   num_run_sizes = 0
   run_sizes = {}

   num_kernels = 0
   kernels = {}
   include_kernels = {}
   exclude_kernels = {}

   num_variants = 0
   variants = {}
   variant_colors = {}
   include_variants = {}
   exclude_variants = {}

   num_tunings = 0
   tunings = {}
   tuning_formats = {}
   include_tunings = {}
   exclude_tunings = {}

   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   info_axes = { "sweep_dir_name": 0, 0: "sweep_dir_name",
                 "run_size": 1,       1: "run_size",
                 "kernel_index": 2,   2: "kernel_index", }

   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   #           variant index - for variants
   #             tuning index - for tunings
   data_axes = { "sweep_dir_name": 0, 0: "sweep_dir_name",
                 "run_size": 1,       1: "run_size",
                 "kernel_index": 2,   2: "kernel_index",
                 "variant_index": 3,  3: "variant_index",
                 "tuning_index": 4,   4: "tuning_index", }

   # multi-dimensional array structured like data but missing some dimensions
   #     directory name - platform, compiler, etc
   #       kernel index - for kernels
   #         variant index - for variants
   #           tuning index - for tunings
   run_size_reduced_axes = { "sweep_dir_name": 0, 0: "sweep_dir_name",
                             "kernel_index": 1,   1: "kernel_index",
                             "variant_index": 2,  2: "variant_index",
                             "tuning_index": 3,   3: "tuning_index", }

   data_model_kind = "time(s)"

   def DataTreeIterator0(self, data_tree):
      assert(data_tree.num_axes == 0)
      yield data_tree.data

   def DataTreeIterator1(self, data_tree):
      assert(data_tree.num_axes == 1)
      assert(data_tree.data)
      for k0 in data_tree.data.keys():
         yield (k0,)

   def DataTreeIterator2(self, data_tree):
      assert(data_tree.num_axes == 2)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1 in v0.keys():
            yield (k0, k1,)

   def DataTreeIterator3(self, data_tree):
      assert(data_tree.num_axes == 3)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2 in v1.keys():
               yield (k0, k1, k2,)

   def DataTreeIterator4(self, data_tree):
      assert(data_tree.num_axes == 4)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3 in v2.keys():
                  yield (k0, k1, k2, k3,)

   def DataTreeIterator5(self, data_tree):
      assert(data_tree.num_axes == 5)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  for k4 in v3.keys():
                     yield (k0, k1, k2, k3, k4,)

   class DataTree:

      def __init__(self, kind, label, type, axes, args=None, func=None):
         self.kind = kind
         self.label = label
         self.type = type
         self.axes = axes
         self.num_axes = len(self.axes) / 2

         self.args = args
         self.func = func

         if self.args:
            self.num_args = len(self.args) / 2

         self.data = None

      def __iter__(self):
         if self.num_axes == 0:
            return Data.DataTreeIterator0(self)
         elif self.num_axes == 1:
            return Data.DataTreeIterator1(self)
         elif self.num_axes == 2:
            return Data.DataTreeIterator2(self)
         elif self.num_axes == 3:
            return Data.DataTreeIterator3(self)
         elif self.num_axes == 4:
            return Data.DataTreeIterator4(self)
         elif self.num_axes == 5:
            return Data.DataTreeIterator5(self)
         else:
            raise ValueError

      def printData(self):
         if self.data:
            print("printData {}:".format(self.kind))
         else:
            print("printData {}: empty".format(self.kind))
            return

         if self.type == "info":
            for sweep_dir_name, sweep_info in self.data.items():
               for run_size, run_info in sweep_info.items():
                  for kernel_index, val in run_info.items():
                     kernel_name = Data.kernels[kernel_index]
                     print("{} {} {} {}".format(sweep_dir_name, run_size, kernel_name, val))

         elif self.type == "data" or \
              self.type == "computed":
            for sweep_dir_name, sweep_data in self.data.items():
               for run_size, run_data in sweep_data.items():
                  for kernel_index, kernel_data in run_data.items():
                     kernel_name = Data.kernels[kernel_index]
                     for variant_index, variant_data in kernel_data.items():
                        variant_name = Data.variants[variant_index]
                        for tuning_index, val in variant_data.items():
                           tuning_name = Data.tunings[tuning_index]
                           print("{} {} {} {} {} {}".format(sweep_dir_name, run_size, kernel_name, variant_name, tuning_name, val))

         elif self.type == "run_size_reduced":
            for sweep_dir_name, sweep_data in self.data.items():
               for kernel_index, kernel_data in sweep_data.items():
                  kernel_name = Data.kernels[kernel_index]
                  for variant_index, variant_data in kernel_data.items():
                     variant_name = Data.variants[variant_index]
                     for tuning_index, val in variant_data.items():
                        tuning_name = Data.tunings[tuning_index]
                        print("{} {} {} {} {}".format(sweep_dir_name, kernel_name, variant_name, tuning_name, val))

         else:
            raise NameError("Can not print type {}".format(self.type))

   class DataTreeTemplate:

      def __init__(self, kind_template, label_template, type, axes, arg_templates, func):
         self.kind_template = kind_template
         self.label_template = label_template
         self.type = type
         self.axes = axes
         self.arg_templates = arg_templates
         self.func = func

      def getKind(self, template_args):
         return self.kind_template.format(*template_args)

      def getLabel(self, template_args):
         arg_labels = [Data.kinds[arg_kind].label for arg_kind in template_args]
         return self.label_template.format(*arg_labels)

      def getArgs(self, template_args):
         return [ arg.format(*template_args) for arg in self.arg_templates ]

      def makeDataTree(self, template_args):
         kind = self.getKind(template_args)
         label = self.getLabel(template_args)
         args = self.getArgs(template_args)
         return Data.DataTree(kind, label, self.type, self.axes, args, self.func)

   # has info derivable from first kind "time(s)" which is read from files
   kinds = { "Problem size":   DataTree("Problem size",   "Problem size", "info", info_axes),
             "Reps":           DataTree("Reps",           "Reps",         "info", info_axes),
             "Iterations/rep": DataTree("Iterations/rep", "Iterations",   "info", info_axes),
             "Kernels/rep":    DataTree("Kernels/rep",    "Kernels",      "info", info_axes),
             "Bytes/rep":      DataTree("Bytes/rep",      "Bytes",        "info", info_axes),
             "FLOPS/rep":      DataTree("FLOPS/rep",      "FLOPS",        "info", info_axes),

             "time(s)": DataTree("time(s)", "time(s)", "data", data_axes),
             "time(ms)": DataTree("time(ms)", "time(ms)", "computed", data_axes, ["time(s)"], lambda t: t * 1000.0),
             "time(us)": DataTree("time(us)", "time(us)", "computed", data_axes, ["time(s)"], lambda t: t * 1000000.0),
             "time(ns)": DataTree("time(ns)", "time(ns)", "computed", data_axes, ["time(s)"], lambda t: t * 1000000000.0),

             "time/rep(s)": DataTree("time/rep(s)", "time(s)", "computed", data_axes, ["time(s)", "Reps"], lambda t, r: t / r),
             "time/rep(ms)": DataTree("time/rep(ms)", "time(ms)", "computed", data_axes, ["time/rep(s)"], lambda tpr: tpr * 1000.0),
             "time/rep(us)": DataTree("time/rep(us)", "time(us)", "computed", data_axes, ["time/rep(s)"], lambda tpr: tpr * 1000000.0),
             "time/rep(ns)": DataTree("time/rep(ns)", "time(ns)", "computed", data_axes, ["time/rep(s)"], lambda tpr: tpr * 1000000000.0),

             "time/it(s)": DataTree("time/it(s)", "time(s)", "computed", data_axes, ["time/rep(s)", "Iterations/rep"], lambda tpr, ipr: tpr / ipr),
             "time/it(ms)": DataTree("time/it(ms)", "time(ms)", "computed", data_axes, ["time/it(s)"], lambda tpi: tpi * 1000.0),
             "time/it(us)": DataTree("time/it(us)", "time(us)", "computed", data_axes, ["time/it(s)"], lambda tpi: tpi * 1000000.0),
             "time/it(ns)": DataTree("time/it(ns)", "time(ns)", "computed", data_axes, ["time/it(s)"], lambda tpi: tpi * 1000000000.0),

             "time/kernel(s)": DataTree("time/kernel(s)", "time(s)", "computed", data_axes, ["time/rep(s)", "Kernels/rep"], lambda tpr, kpr: tpr / kpr),
             "time/kernel(ms)": DataTree("time/kernel(ms)", "time(ms)", "computed", data_axes, ["time/kernel(s)"], lambda tpk: tpk * 1000.0),
             "time/kernel(us)": DataTree("time/kernel(us)", "time(us)", "computed", data_axes, ["time/kernel(s)"], lambda tpk: tpk * 1000000.0),
             "time/kernel(ns)": DataTree("time/kernel(ns)", "time(ns)", "computed", data_axes, ["time/kernel(s)"], lambda tpk: tpk * 1000000000.0),

             "throughput(Problem size/s)": DataTree("throughput(Problem size/s)", "throughput(Problem size/s)", "computed", data_axes, ["time/rep(s)", "Problem size"], lambda tpr, ps: ps / tpr),
             "throughput(Problem size/ms)": DataTree("throughput(Problem size/ms)", "throughput(Problem size/ms)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000.0),
             "throughput(Problem size/us)": DataTree("throughput(Problem size/us)", "throughput(Problem size/us)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000000.0),
             "throughput(Problem size/ns)": DataTree("throughput(Problem size/ns)", "throughput(Problem size/ns)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000000000.0),
             "throughput(KProblem size/s)": DataTree("throughput(KProblem size/s)", "throughput(KProblem size/s)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000.0),
             "throughput(MProblem size/s)": DataTree("throughput(MProblem size/s)", "throughput(MProblem size/s)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000000.0),
             "throughput(GProblem size/s)": DataTree("throughput(GProblem size/s)", "throughput(GProblem size/s)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000000000.0),
             "throughput(TProblem size/s)": DataTree("throughput(TProblem size/s)", "throughput(TProblem size/s)", "computed", data_axes, ["throughput(Problem size/s)"], lambda thr: thr / 1000000000000.0),

             "bandwidth(B/s)": DataTree("bandwidth(B/s)", "bandwidth(B/s)", "computed", data_axes, ["time/rep(s)", "Bytes/rep"], lambda tpr, bpr: bpr / tpr),
             "bandwidth(KB/s)": DataTree("bandwidth(KB/s)", "bandwidth(KB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1000.0),
             "bandwidth(MB/s)": DataTree("bandwidth(MB/s)", "bandwidth(MB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1000000.0),
             "bandwidth(GB/s)": DataTree("bandwidth(GB/s)", "bandwidth(GB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1000000000.0),
             "bandwidth(TB/s)": DataTree("bandwidth(TB/s)", "bandwidth(TB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1000000000000.0),
             "bandwidth(KiB/s)": DataTree("bandwidth(KiB/s)", "bandwidth(KiB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1024.0),
             "bandwidth(MiB/s)": DataTree("bandwidth(MiB/s)", "bandwidth(MiB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1048576.0),
             "bandwidth(GiB/s)": DataTree("bandwidth(GiB/s)", "bandwidth(GiB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1073741824.0),
             "bandwidth(TiB/s)": DataTree("bandwidth(TiB/s)", "bandwidth(TiB/s)", "computed", data_axes, ["bandwidth(B/s)"], lambda bps: bps / 1099511627776.0),

             "FLOPS": DataTree("FLOPS", "FLOPS", "computed", data_axes, ["time/rep(s)", "FLOPS/rep"], lambda tpr, fpr: fpr / tpr),
             "KFLOPS": DataTree("KFLOPS", "KFLOPS", "computed", data_axes, ["FLOPS"], lambda fps: fps / 1000.0),
             "MFLOPS": DataTree("MFLOPS", "MFLOPS", "computed", data_axes, ["FLOPS"], lambda fps: fps / 1000000.0),
             "GFLOPS": DataTree("GFLOPS", "GFLOPS", "computed", data_axes, ["FLOPS"], lambda fps: fps / 1000000000.0),
             "TFLOPS": DataTree("TFLOPS", "TFLOPS", "computed", data_axes, ["FLOPS"], lambda fps: fps / 1000000000000.0),

      }

   kind_templates = {
             "first": DataTreeTemplate("first<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], first),
             "last": DataTreeTemplate("last<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], last),

             "min": DataTreeTemplate("min<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], min),
             "max": DataTreeTemplate("max<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], max),
             "sum": DataTreeTemplate("sum<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], sum),
             "avg": DataTreeTemplate("avg<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], avg),
             "stddev": DataTreeTemplate("stddev<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], stddev),
             "relstddev": DataTreeTemplate("relstddev<{0}>", "{0}", "run_size_reduced", run_size_reduced_axes, ["{0}"], relstddev),

             "_LR": DataTreeTemplate("_LR<{0}>", "intercept, slope, correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["Problem size", "{0}"], linearRegression),
             "LR_intercept": DataTreeTemplate("LR_intercept<{0}>", "intercept", "run_size_reduced", run_size_reduced_axes, ["_LR<{0}>"], lambda lr: lr[0]),
             "LR_slope": DataTreeTemplate("LR_slope<{0}>", "slope", "run_size_reduced", run_size_reduced_axes, ["_LR<{0}>"], lambda lr: lr[1]),
             "LR_correlationCoefficient": DataTreeTemplate("LR_correlationCoefficient<{0}>", "correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["_LR<{0}>"], lambda lr: lr[2]),
             "LR": DataTreeTemplate("LR<{0}>", "{0}", "computed", data_axes, ["Problem size", "_LR<{0}>"], eval_linearRegression),

             "_LR_log": DataTreeTemplate("_LR_log<{0}>", "intercept, slope, correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["Problem size", "{0}"], linearRegression_loglog),
             "LR_log_intercept": DataTreeTemplate("LR_log_intercept<{0}>", "intercept", "run_size_reduced", run_size_reduced_axes, ["_LR_log<{0}>"], lambda lr: lr[0]),
             "LR_log_slope": DataTreeTemplate("LR_log_slope<{0}>", "slope", "run_size_reduced", run_size_reduced_axes, ["_LR_log<{0}>"], lambda lr: lr[1]),
             "LR_log_correlationCoefficient": DataTreeTemplate("LR_log_correlationCoefficient<{0}>", "correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["_LR_log<{0}>"], lambda lr: lr[2]),
             "LR_log": DataTreeTemplate("LR_log<{0}>", "{0}", "computed", data_axes, ["Problem size", "_LR_log<{0}>"], eval_linearRegression_loglog),

             "_LR2": DataTreeTemplate("_LR2<{0}>", "intercept, slope, correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["Problem size", "{0}"], segmented_linearRegression),
             "LR2_intercept": DataTreeTemplate("LR2_intercept<{0}>", "intercept", "run_size_reduced", run_size_reduced_axes, ["_LR2<{0}>"], lambda lr: lr[0]),
             "LR2_slope": DataTreeTemplate("LR2_slope<{0}>", "slope", "run_size_reduced", run_size_reduced_axes, ["_LR2<{0}>"], lambda lr: lr[1]),
             "LR2_correlationCoefficient": DataTreeTemplate("LR2_correlationCoefficient<{0}>", "correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["_LR2<{0}>"], lambda lr: lr[2]),
             "LR2": DataTreeTemplate("LR2<{0}>", "{0}", "computed", data_axes, ["Problem size", "_LR2<{0}>"], eval_segmented_linearRegression),

             "_LR2_log": DataTreeTemplate("_LR2_log<{0}>", "intercept, slope, correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["Problem size", "{0}"], segmented_linearRegression_loglog),
             "LR2_log_intercept": DataTreeTemplate("LR2_log_intercept<{0}>", "intercept", "run_size_reduced", run_size_reduced_axes, ["_LR2_log<{0}>"], lambda lr: lr[0]),
             "LR2_log_slope": DataTreeTemplate("LR2_log_slope<{0}>", "slope", "run_size_reduced", run_size_reduced_axes, ["_LR2_log<{0}>"], lambda lr: lr[1]),
             "LR2_log_correlationCoefficient": DataTreeTemplate("LR2_log_correlationCoefficient<{0}>", "correlation coefficient", "run_size_reduced", run_size_reduced_axes, ["_LR2_log<{0}>"], lambda lr: lr[2]),
             "LR2_log": DataTreeTemplate("LR2_log<{0}>", "{0}", "computed", data_axes, ["Problem size", "_LR2_log<{0}>"], eval_segmented_linearRegression_loglog),

      }

   def compute_data(kind):
      if not kind in Data.kinds:
         raise NameError("Unknown data kind {}".format(kind))

      if Data.kinds[kind].data:
         return # already calculated


      if Data.kinds[kind].type == "computed":

         if not (Data.kinds[kind].args and Data.kinds[kind].func):
            raise NameError("Computing data is not supported for kind {0}".format(kind))

         compute_args = Data.kinds[kind].args
         compute_func = Data.kinds[kind].func

         for arg_kind in compute_args:
            # calculate data for arg_kind
            Data.compute(arg_kind)

         if (not Data.data_model_kind in Data.kinds) or (not Data.kinds[Data.data_model_kind].data):
            raise NameError("Model data not available {0}, no args".format(Data.data_model_kind))

         Data.kinds[kind].data = {}
         for sweep_dir_name, model_sweep_data in Data.kinds[Data.data_model_kind].data.items():
            Data.kinds[kind].data[sweep_dir_name] = {}
            for run_size, model_run_data in model_sweep_data.items():
               Data.kinds[kind].data[sweep_dir_name][run_size] = {}
               for kernel_index, model_kernel_data in model_run_data.items():
                  kernel_name = Data.kernels[kernel_index]
                  Data.kinds[kind].data[sweep_dir_name][run_size][kernel_index] = {}
                  for variant_index, model_variant_data in model_kernel_data.items():
                     variant_name = Data.variants[variant_index]
                     Data.kinds[kind].data[sweep_dir_name][run_size][kernel_index][variant_index] = {}
                     for tuning_index, model_val in model_variant_data.items():
                        tuning_name = Data.tunings[tuning_index]

                        args_val = ()
                        for arg_kind in compute_args:
                           if Data.kinds[arg_kind].type == "info":
                              arg_val = Data.kinds[arg_kind].data[sweep_dir_name][run_size][kernel_index]
                           elif Data.kinds[arg_kind].type == "data" or Data.kinds[arg_kind].type == "computed":
                              arg_val = Data.kinds[arg_kind].data[sweep_dir_name][run_size][kernel_index][variant_index][tuning_index]
                           elif Data.kinds[arg_kind].type == "run_size_reduced":
                              arg_val = Data.kinds[arg_kind].data[sweep_dir_name][kernel_index][variant_index][tuning_index]
                           else:
                              raise NameError("Invalid data kind {0}".format(arg_kind))
                           args_val = args_val + (arg_val,)

                        val = compute_func(*args_val)
                        Data.kinds[kind].data[sweep_dir_name][run_size][kernel_index][variant_index][tuning_index] = val

      elif Data.kinds[kind].type == "run_size_reduced":

         if not (Data.kinds[kind].func and Data.kinds[kind].args):
            raise NameError("Reducing data is not supported for kind {0}".format(kind))

         reduce_args = Data.kinds[kind].args
         reduce_func = Data.kinds[kind].func

         for arg_kind in reduce_args:
            # calculate data for arg_kind
            Data.compute(arg_kind)

         if (not Data.data_model_kind in Data.kinds) or (not Data.kinds[Data.data_model_kind].data):
            raise NameError("Model data not available {0}, no args".format(Data.data_model_kind))

         Data.kinds[kind].data = {}
         for sweep_dir_name, model_sweep_data in Data.kinds[Data.data_model_kind].data.items():
            Data.kinds[kind].data[sweep_dir_name] = {}
            for run_size, model_run_data in model_sweep_data.items():
               for kernel_index, model_kernel_data in model_run_data.items():
                  kernel_name = Data.kernels[kernel_index]
                  if not kernel_index in Data.kinds[kind].data[sweep_dir_name]:
                     Data.kinds[kind].data[sweep_dir_name][kernel_index] = {}
                  for variant_index, model_variant_data in model_kernel_data.items():
                     variant_name = Data.variants[variant_index]
                     if not variant_index in Data.kinds[kind].data[sweep_dir_name][kernel_index]:
                        Data.kinds[kind].data[sweep_dir_name][kernel_index][variant_index] = {}
                     for tuning_index, model_val in model_variant_data.items():
                        tuning_name = Data.tunings[tuning_index]

                        if not tuning_index in Data.kinds[kind].data[sweep_dir_name][kernel_index][variant_index]:
                           args_val = ()
                           for arg_kind in reduce_args:
                              if Data.kinds[arg_kind].type == "info":
                                 arg_val = []
                              elif Data.kinds[arg_kind].type == "data" or Data.kinds[arg_kind].type == "computed":
                                 arg_val = []
                              elif Data.kinds[arg_kind].type == "run_size_reduced":
                                 arg_val = Data.kinds[arg_kind].data[sweep_dir_name][kernel_index][variant_index][tuning_index]
                              else:
                                 raise NameError("Invalid data kind {0}".format(arg_kind))
                              args_val = args_val + (arg_val,)
                           Data.kinds[kind].data[sweep_dir_name][kernel_index][variant_index][tuning_index] = args_val
                        else:
                           args_val = Data.kinds[kind].data[sweep_dir_name][kernel_index][variant_index][tuning_index]

                        args_idx = 0
                        for arg_kind in reduce_args:
                           if Data.kinds[arg_kind].type == "info":
                              args_val[args_idx].append(Data.kinds[arg_kind].data[sweep_dir_name][run_size][kernel_index])
                           elif Data.kinds[arg_kind].type == "data" or Data.kinds[arg_kind].type == "computed":
                              args_val[args_idx].append(Data.kinds[arg_kind].data[sweep_dir_name][run_size][kernel_index][variant_index][tuning_index])
                           elif Data.kinds[arg_kind].type == "run_size_reduced":
                              pass
                           else:
                              raise NameError("Invalid data kind {0}".format(arg_kind))
                           args_idx += 1

         for sweep_dir_name, sweep_data in Data.kinds[kind].data.items():
            for kernel_index, kernel_data in sweep_data.items():
               for variant_index, variant_data in kernel_data.items():
                  for tuning_index, args_val in variant_data.items():
                     Data.kinds[kind].data[sweep_dir_name][kernel_index][variant_index][tuning_index] = reduce_func(*args_val)

      else:
         raise NameError("Unknown kind type {}".format(Data.kinds[kind].type))

   def compute_templated_data(kind_template, template_args):
      if kind_template in Data.kind_templates:
         kind = Data.kind_templates[kind_template].getKind(template_args)
         if not kind in Data.kinds:
            # compute args first to ensure arg kinds exist
            for arg_kind in Data.kind_templates[kind_template].getArgs(template_args):
               Data.compute(arg_kind)
            Data.kinds[kind] = Data.kind_templates[kind_template].makeDataTree(template_args)
            Data.compute(kind)
      else:
         raise NameError("Unkown kind template {}".format(kind_template))

   def kind_template_scan(kind):

      template_args = None

      template_start_idx = kind.find("<")
      template_end_idx = kind.rfind(">")

      if template_start_idx == -1 or template_end_idx == -1:
         return kind, template_args

      kind_template = kind[:template_start_idx]
      template_args = kind[template_start_idx+1:template_end_idx].split(",")

      for i in range(0,len(template_args)):
         template_args[i] = template_args[i].strip()

      return (kind_template, template_args)

   def compute(kind):
      if kind in Data.kinds:
         if not Data.kinds[kind].data:
            Data.compute_data(kind)
         return

      kind_template, template_args = Data.kind_template_scan(kind)
      if kind_template in Data.kind_templates:
         Data.compute_templated_data(kind_template, template_args)
         return

      raise NameError("Unknown data kind {}".format(kind))



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
               if not info_kind in Data.kinds:
                  # add new kind to global data
                  print("Unknown kernel info {0}".format(info_kind))
                  Data.kinds[info_kind] = Data.DataTree(info_kind, "info", Data.info_axes)
               if info_kind in c_to_info_kinds:
                  print("Repeated kernel info {0}".format(info_kind))
                  sys.exit(1)
               if not Data.kinds[info_kind].data:
                  # add data to kind
                  Data.kinds[info_kind].data = {}
               if not sweep_dir_name in Data.kinds[info_kind].data:
                  # add new sweep to global data
                  Data.kinds[info_kind].data[sweep_dir_name] = {}
               if run_size in Data.kinds[info_kind].data[sweep_dir_name]:
                  print("Repeated kernel size {0} in {1}".format(sweep_dir_name, run_size))
                  sys.exit(1)
               else:
                  # add new size to global data
                  Data.kinds[info_kind].data[sweep_dir_name][run_size] = {}
               # make map of columns to names
               c_to_info_kinds[c] = info_kind
               c_to_info_kinds[info_kind] = c
         elif not ignore:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in Data.kernels:
               kernel_index = Data.kernels[kernel_name]
            elif (len(Data.include_kernels) == 0 or kernel_name in Data.include_kernels) and (not kernel_name in Data.exclude_kernels):
               # add kernel to global list
               kernel_index = Data.num_kernels
               Data.num_kernels += 1
               Data.kernels[kernel_name]  = kernel_index
               Data.kernels[kernel_index] = kernel_name
            else:
               continue # skip this kernel

            for c in range(1, len(row)):
               info_kind = c_to_info_kinds[c]
               try:
                  # add data to global structure
                  val = int(row[c].strip())
                  # print(kernel_index, kernel_name, info_kind, val)
                  Data.kinds[info_kind].data[sweep_dir_name][run_size][kernel_index] = val
               except ValueError:
                  pass # could not convert data to int


def read_timing_file(sweep_dir_name, sweep_subdir_timing_file_path, run_size):
   # print(sweep_dir_name, sweep_subdir_timing_file_path, run_size)
   with open(sweep_subdir_timing_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      data_kind = g_timing_file_kind
      if not data_kind in Data.kinds:
         raise NameError("Unknown kind {}".format(data_kind))
      if not Data.kinds[data_kind].data:
         Data.kinds[data_kind].data = {}
      if not sweep_dir_name in Data.kinds[data_kind].data:
         Data.kinds[data_kind].data[sweep_dir_name] = {}
      if not run_size in Data.kinds[data_kind].data[sweep_dir_name]:
         Data.kinds[data_kind].data[sweep_dir_name][run_size] = {}
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
                  if variant_name in Data.variants:
                     pass
                  elif (len(Data.include_variants) == 0 or variant_name in Data.include_variants) and (not variant_name in Data.exclude_variants):
                     variant_index = Data.num_variants
                     Data.num_variants += 1
                     Data.variants[variant_name]  = variant_index
                     Data.variants[variant_index] = variant_name
                     if variant_name in g_known_variants:
                        variant_color = g_known_variants[variant_name]["color"]
                        Data.variant_colors[variant_name] = variant_color
                        Data.variant_colors[variant_index] = variant_color
                     else:
                        print("Unknown variant {0}".format(variant_name))
                        sys.exit(1)
                  else:
                     Data.variants[variant_name]  = -1

                  c_to_variant_index[c] = Data.variants[variant_name]
            elif len(c_to_tuning_index) == 0:
               for c in range(1, len(row)):
                  tuning_name = row[c].strip()
                  if tuning_name in Data.tunings:
                     pass
                  elif (len(Data.include_tunings) == 0 or tuning_name in Data.include_tunings) and (not tuning_name in Data.exclude_tunings):
                     tuning_index = Data.num_tunings
                     Data.num_tunings += 1
                     Data.tunings[tuning_name]  = tuning_index
                     Data.tunings[tuning_index] = tuning_name
                     if tuning_name in g_known_tunings:
                        tuning_format = g_known_tunings[tuning_name]["format"]
                        Data.tuning_formats[tuning_name] = tuning_format
                        Data.tuning_formats[tuning_index] = tuning_format
                     else:
                        print("Unknown tuning {0}".format(tuning_name))
                        sys.exit(1)
                  else:
                     Data.tunings[tuning_name] = -1
                  c_to_tuning_index[c] = Data.tunings[tuning_name]
            else:
               print("Unknown row {0}".format(row))
               sys.exit(1);
         elif len(c_to_variant_index) > 0 and len(c_to_tuning_index) > 0:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in Data.kernels:
               kernel_index = Data.kernels[kernel_name]
            else:
               continue # skip kernel

            Data.kinds[data_kind].data[sweep_dir_name][run_size][kernel_index] = {}
            for c in range(1, len(row)):
               variant_index = c_to_variant_index[c]
               tuning_index = c_to_tuning_index[c]
               if variant_index < 0 or tuning_index < 0:
                  continue # ignore data
               try:
                  val = float(row[c].strip())
                  # print(kernel_index, kernel_name, variant_index, tuning_index, data_kind, val)
                  if not variant_index in Data.kinds[data_kind].data[sweep_dir_name][run_size][kernel_index]:
                     Data.kinds[data_kind].data[sweep_dir_name][run_size][kernel_index][variant_index] = {}
                  Data.kinds[data_kind].data[sweep_dir_name][run_size][kernel_index][variant_index][tuning_index] = val
               except ValueError:
                  pass # could not convert data to float


def get_plot_data(kind, kernel):

   if not kind in Data.kinds:
      raise NameError("Unknown kind {}".format(kind))

   kernel_index = kernel
   if isinstance(kernel, str):
      if kernel in Data.kernels:
         kernel_index = Data.kernels[kernel]
      else:
         raise NameError("Unknown kernel {}".format(kernel))
   elif isinstance(kernel, int):
      kernel_index = kernel
   else:
      raise NameError("Unknown kernel {}".format(kernel))

   if not kernel_index in Data.kernels:
      raise NameError("Unknown kernel {}".format(kernel_index))
   kernel_name = Data.kernels[kernel_index]

   data = {}

   kind_meta = Data.kinds[kind]
   if kind_meta.type == "info":

      if not kind in Data.kinds:
         raise NameError("Unknown info kind {}".format(kind))
      kind_info = Data.kinds[kind].data

      for sweep_dir_name, sweep_info in kind_info.items():

         data[sweep_dir_name] = {}
         data[sweep_dir_name][kind] = {"type": "info",
                                       "data": [] }

         for run_size, run_info in sweep_info.items():

            if not kernel_index in run_info:
               raise NameError("Unknown info kernel_index {}".format(kernel_index))

            val = run_info[kernel_index]
            data[sweep_dir_name][kind]["data"].append(val)

   elif kind_meta.type == "data" or kind_meta.type == "computed":

      if not kind in Data.kinds:
         raise NameError("Unknown data kind {}".format(kind))
      kind_data = Data.kinds[kind].data

      for sweep_dir_name, sweep_data in kind_data.items():

         data[sweep_dir_name] = {}

         for run_size, run_data in sweep_data.items():

            if not kernel_index in run_data:
               raise NameError("Unknown info kernel_index {}".format(kernel_index))

            kernel_data = run_data[kernel_index]

            for variant_index, variant_data in kernel_data.items():
               variant_name = Data.variants[variant_index]
               for tuning_index, val in variant_data.items():
                  tuning_name = Data.tunings[tuning_index]

                  data_name = "{}-{}".format(variant_name, tuning_name)

                  if not data_name in data[sweep_dir_name]:
                     data[sweep_dir_name][data_name] = {"type": "data",
                                                        "variant": variant_index,
                                                        "tuning": tuning_index,
                                                        "data": [] }

                  data[sweep_dir_name][data_name]["data"].append(val)

   elif kind_meta.type == "run_size_reduced":

      if not kind in Data.kinds:
         raise NameError("Unknown data kind {}".format(kind))
      kind_data = Data.kinds[kind].data

      for sweep_dir_name, sweep_data in kind_data.items():

         data[sweep_dir_name] = {}

         if not kernel_index in sweep_data:
            raise NameError("Unknown info kernel_index {}".format(kernel_index))

         kernel_data = sweep_data[kernel_index]

         for variant_index, variant_data in kernel_data.items():
            variant_name = Data.variants[variant_index]
            for tuning_index, val in variant_data.items():
               tuning_name = Data.tunings[tuning_index]

               data_name = "{}-{}".format(variant_name, tuning_name)

               if not data_name in data[sweep_dir_name]:
                  data[sweep_dir_name][data_name] = {"type": "data",
                                                     "variant": variant_index,
                                                     "tuning": tuning_index,
                                                     "data": [] }

               data[sweep_dir_name][data_name]["data"].append(val)

   else:
      raise NameError("Unknown kind {} type {}".format(kind, kind_meta.type))

   return data


def plot_data_problem_sizes(outputfile_name, ykinds):

   ylabel = None
   yscale = "log"
   ylim = None

   xkind = "Problem size"
   xlabel = Data.kinds[xkind].label
   xscale = "log"
   xlim = None

   for ykind in ykinds:
      if not ykind in Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not ylabel:
         ylabel = Data.kinds[ykind].label
      elif ylabel != Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([Data.kinds[_ykind].label for _ykind in ykinds]))

   for kernel_index in range(0, Data.num_kernels):
      kernel_name = Data.kernels[kernel_index]

      for ykind in ykinds:

         xaxes = get_plot_data(xkind, kernel_index)
         yaxes = get_plot_data(ykind, kernel_index)

         for sweep_index in range(0, Data.num_sweeps):
            sweep_dir_name = Data.sweeps[sweep_index]
            sweep_marker = Data.sweep_markers[sweep_index]

            if not sweep_dir_name in xaxes:
               raise NameError("Unknown sweep_dir_name {}".format(sweep_dir_name))

            for data_name, ydata in yaxes[sweep_dir_name].items():

               yname = "{} {} {}".format(Data.kinds[ykind].kind, data_name, sweep_dir_name)
               yformat = "{}-".format(sweep_marker)
               ycolor = (0.0, 0.0, 0.0, 1.0)
               yaxis = ydata["data"]

               if ydata["type"] == "data":
                  variant_index = ydata["variant"]
                  tuning_index = ydata["tuning"]
                  ycolor = Data.variant_colors[variant_index]
                  yformat = "{}{}".format(sweep_marker, Data.tuning_formats[tuning_index])

               if data_name in xaxes[sweep_dir_name]:
                  xaxis = xaxes[sweep_dir_name][data_name]["data"]
               elif xkind in xaxes[sweep_dir_name]:
                  xaxis = xaxes[sweep_dir_name][xkind]["data"]
               else:
                  raise NameError("Unknown xaxis for {}".format(data_name))

               plt.plot(xaxis,yaxis,yformat,color=ycolor,label=yname)

      fname = "{}_{}.png".format(outputfile_name, kernel_name)
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

def plot_data_kernels(outputfile_name, ykinds):

   gname = None

   xlabel = "Kernel"
   xscale = None
   xlim = None

   ylabel = None
   yscale = None
   ylim = None
   ywidth = None

   for ykind in ykinds:
      if gname:
         gname = "{} {}".format(gname, ykind)
      else:
         gname = "{}".format(ykind)
      if not ykind in Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not ylabel:
         ylabel = Data.kinds[ykind].label
      elif ylabel != Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([Data.kinds[_ykind].label for _ykind in ykinds]))

   kernel_data = { "kernel_names": [],
                   "kernel_centers": [],
                   "ynames": {},
                   "ydata": {}, }

   for kernel_index in range(0, Data.num_kernels):
      kernel_name = Data.kernels[kernel_index]

      kernel_data["kernel_names"].append(kernel_name)
      kernel_data["kernel_centers"].append(kernel_index)

      for ykind in ykinds:

         yaxes = get_plot_data(ykind, kernel_index)

         for sweep_index in range(0, Data.num_sweeps):
            sweep_dir_name = Data.sweeps[sweep_index]

            if not sweep_dir_name in yaxes:
               raise NameError("Unknown sweep_dir_name {}".format(sweep_dir_name))

            for data_name, ydata in yaxes[sweep_dir_name].items():
               assert(len(ydata["data"]) == 1)

               yname = "{} {}".format(data_name, sweep_dir_name)
               if len(ykinds) > 1:
                  yname = "{} {}".format(Data.kinds[ykind].kind, yname)

               ycolor = (0.0, 0.0, 0.0, 1.0)

               if ydata["type"] == "data":
                  variant_index = ydata["variant"]
                  # tuning_index = ydata["tuning"]
                  ycolor = Data.variant_colors[variant_index]

               if not yname in kernel_data["ynames"]:
                  kernel_data["ynames"][yname] = len(kernel_data["ynames"])
                  kernel_data["ydata"][yname] = { "color": ycolor,
                                                  "data": [], }

               # pad with 0s if find missing data
               while len(kernel_data["ydata"][yname]["data"])+1 < len(kernel_data["kernel_names"]):
                  kernel_data["ydata"][yname]["data"].append(0.0)

               kernel_data["ydata"][yname]["data"].append(ydata["data"][0])

   num_xticks = len(kernel_data["kernel_centers"])
   plt.figure(figsize=(max(num_xticks*0.5, 4), 6,))

   y_n = len(kernel_data["ydata"])
   ywidth = 1.0 / (y_n+1)
   for yname in kernel_data["ynames"]:

      y_i = kernel_data["ynames"][yname]
      ydata = kernel_data["ydata"][yname]

      xaxis = [c + (y_i+1)/(y_n+1) - 0.5 for c in kernel_data["kernel_centers"]]
      yaxis = ydata["data"]

      # pad with 0s if find missing data
      while len(yaxis) < len(kernel_data["kernel_names"]):
         yaxis.append(0.0)

      ycolor = ydata["color"]

      plt.bar(xaxis,yaxis,label=yname,width=ywidth,color=ycolor) # ,edgecolor="grey")

   fname = "{}.png".format(outputfile_name)
   if not gname:
      gname = "{}".format("bar")

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
   plt.legend()
   plt.grid(True)

   plt.savefig(fname, dpi=150.0, bbox_inches="tight")
   plt.clf()

def plot_data(outputfile_name, ykinds):
   print("plotting {} {}".format(outputfile_name, ykinds))

   func = None
   for ykind in ykinds:
      if Data.kinds[ykind].type == "run_size_reduced":
         func = plot_data_kernels
      else:
         func = plot_data_problem_sizes

   func(outputfile_name, ykinds)




def main(argv):
   sweep_dir_paths = []
   outputfile = "graph"
   help_string = "sweep_graph.py -o <outputfile> <sweepdir1 [sweepdir2 ...]>"
   runinfo_filename = g_runinfo_filename
   timing_filename = g_timing_filename
   print_kinds = []
   graph_kinds = []

   i = 0
   while i < len(argv):
      opt = argv[i]
      if len(opt) == 0:
         print(help_string)
         sys.exit(2)
      elif opt[0] == "-":

         # no arg options
         if opt in ("-h", "--help"):
            print(help_string)
            sys.exit()

         handle_num = 0
         handle_arg = None
         # single arg options
         if opt in ("-o", "--output"):
            handle_num = 1
            def fo(arg):
               nonlocal outputfile
               outputfile = arg
            handle_arg = fo
         # multi arg options
         elif opt in ("-p", "--print"):
            handle_num = -1
            def p(arg):
               print_kinds.append(arg)
            handle_arg = p
         elif opt in ("-g", "--graph"):
            handle_num = -1
            def fg(arg):
               graph_kinds.append(arg)
            handle_arg = fg
         elif opt in ("-k", "--kernels"):
            handle_num = -1
            def fk(arg):
               Data.include_kernels[arg] = arg
            handle_arg = fk
         elif opt in ("-ek", "--exclude-kernels"):
            handle_num = -1
            def fek(arg):
               Data.exclude_kernels[arg] = arg
            handle_arg = fek
         elif opt in ("-v", "--variants"):
            handle_num = -1
            def fv(arg):
               Data.include_variants[arg] = arg
            handle_arg = fv
         elif opt in ("-ev", "--exclude-variants"):
            handle_num = -1
            def fev(arg):
               Data.exclude_variants[arg] = arg
            handle_arg = fev
         elif opt in ("-t", "--tunings"):
            handle_num = -1
            def ft(arg):
               Data.include_tunings[arg] = arg
            handle_arg = ft
         elif opt in ("-et", "--exclude-tunings"):
            handle_num = -1
            def fet(arg):
               Data.exclude_tunings[arg] = arg
            handle_arg = fet

         if handle_num == 0:
            print(help_string)
            sys.exit(2)

         elif handle_num > 0:
            if not i+handle_num < len(argv):
               print("Missing option to {}".format(opt))
               sys.exit(2)
            for h in range(1, handle_num+1):
               arg = argv[i+h]
               if arg[0] == "-":
                  print("Missing option to {}".format(opt))
                  sys.exit(2)
               handle_arg(arg)
            i += handle_num

         else:
            while i+1 < len(argv):
               arg = argv[i+1]
               if arg[0] == "-":
                  break
               handle_arg(arg)
               i += 1

      else:
         sweep_dir_paths.append(opt)
      i += 1

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

               if not str(run_size) in Data.run_sizes:
                  Data.run_sizes[Data.num_run_sizes] = run_size
                  Data.run_sizes[str(run_size)] = Data.num_run_sizes
                  Data.num_run_sizes += 1

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
         if sweep_dir_name in Data.sweeps:
            raise NameError("Repeated sweep_dir_name {}".format(sweep_dir_name))
         Data.sweeps[Data.num_sweeps] = sweep_dir_name
         if Data.num_sweeps >= len(g_markers):
            raise NameError("Ran out of sweep markers for {}".format(sweep_dir_name))
         Data.sweep_markers[Data.num_sweeps] = g_markers[Data.num_sweeps]
         Data.num_sweeps += 1

   kinds_string = ""
   for kindTree in Data.kinds.values():
      kinds_string += ", {}".format(kindTree.kind)
   print("kinds")
   print("  {}".format(kinds_string[2:]))

   kind_templates_string = ""
   for kindTree_template in Data.kind_templates.values():
      kind_templates_string += ", {}".format(kindTree_template.kind_template)
   print("kind_templates")
   print("  {}".format(kind_templates_string[2:]))

   sweeps_string = ""
   for v in range(0, Data.num_sweeps):
      sweeps_string += ", {}".format(Data.sweeps[v])
   print("sweeps")
   print("  {}".format(sweeps_string[2:]))

   run_sizes_string = ""
   for v in range(0, Data.num_run_sizes):
      run_sizes_string += ", {}".format(Data.run_sizes[v])
   print("run_sizes")
   print("  {}".format(run_sizes_string[2:]))

   kernel_string = ""
   for v in range(0, Data.num_kernels):
      kernel_string += ", {}".format(Data.kernels[v])
   print("kernels")
   print("  {}".format(kernel_string[2:]))

   variant_string = ""
   for v in range(0, Data.num_variants):
      variant_string += ", {}".format(Data.variants[v])
   print("variants")
   print("  {}".format(variant_string[2:]))

   tuning_string = ""
   for v in range(0, Data.num_tunings):
      tuning_string += ", {}".format(Data.tunings[v])
   print("tunings")
   print("  {}".format(tuning_string[2:]))

   for kind in print_kinds:
      Data.compute(kind)
      Data.kinds[kind].printData()

   if len(graph_kinds) > 0:
      for kind in graph_kinds:
         Data.compute(kind)
      plot_data(outputfile, graph_kinds)

if __name__ == "__main__":
   main(sys.argv[1:])
