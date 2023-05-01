import math

def make_tuple_str(astr):
   astr = astr.strip()
   if len(astr) < 2 or astr[0] != "(" or astr[len(astr) - 1] != ")":
      return None
   astr = astr[1:len(astr) - 1]
   atup = astr.split(",")
   return tuple((a.strip() for a in atup))


def normalize_color_tuple(t):
   len_t = 0.0
   for i in range(0, len(t)):
      len_t += t[i] * t[i]
   len_t = math.sqrt(len_t)
   new_t = ()
   for i in range(0, len(t)):
      new_t += (t[i] / len_t,)
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
      new_t += (t[i] * factor,)
   return clamp_tuple(new_t)


def make_color_tuple_rgb(r, g, b):
   return (r / 255.0, g / 255.0, b / 255.0)


def make_color_tuple_str(color_str):
   rgb = make_tuple_str(color_str)
   if len(rgb) != 3:
      raise NameError("Expected a tuple of 3 floats in [0-1]")
   r = float(rgb[0].strip())
   g = float(rgb[1].strip())
   b = float(rgb[2].strip())
   return clamp_tuple((r, g, b))


def set_legend_order(labels) -> list:
   print("set_legend_order:" + str(labels))
   lex_order = ['machine', 'execution_model', 'programming_model', 'tuning']
   legend_order = []
   
   machines = []
   programming_models = []
   execution_models = []
   tunings = []
   for label in labels:
      ll = label.split(" ")
      machines.append(ll[0])
      variant = ll[1]
      vs = variant.split("_")
      programming_models.append(vs[0])
      execution_models.append(vs[1])
      tunings.append(ll[2])
   
   machines_min_len = len(min(machines, key=len))
   prg_min_len = len(min(programming_models, key=len))
   exe_min_len = len(min(execution_models, key=len))
   tunings_min_len = len(min(tunings, key=len))
   lex_strings = []
   for i in range(len(machines)):
      machines[i] = machines[i][0:machines_min_len]
      programming_models[i] = programming_models[i][0:prg_min_len]
      execution_models[i] = execution_models[i][0:exe_min_len]
      tunings[i] = tunings[i][0:tunings_min_len]
      lex_string = ""
      for lo in lex_order:
         if lo == 'machine':
            lex_string += machines[i]
         if lo == 'programming_model':
            lex_string += programming_models[i]
         if lo == 'execution_model':
            lex_string += execution_models[i]
         if lo == 'tuning':
            lex_string += tunings[i]
      lex_strings.append([lex_string, i])
   lex_strings.sort()
   
   for x in lex_strings:
      legend_order.append(x[1])
   
   # print(lex_strings)
   # print(legend_order)
   
   return legend_order


#g_timing_filename = "RAJAPerf-timing-Minimum.csv"
g_timing_filename = "RAJAPerf-timing-Average.csv"
g_runinfo_filename = "RAJAPerf-kernels.csv"
g_timing_file_kind = "time(s)"

# Kernels sorted into categories based on performance bottlenecks

g_known_kernel_groups = {
   "bandwidth": {
      "kind": "bandwidth(GiB/s)",
      "kernels": [ "Basic_DAXPY", "Basic_DAXPY_ATOMIC", "Basic_INIT3",
                   "Basic_INIT_VIEW1D", "Basic_INIT_VIEW1D_OFFSET",
                   "Basic_MULADDSUB", "Lcals_DIFF_PREDICT", "Lcals_EOS",
                   "Lcals_FIRST_DIFF", "Lcals_FIRST_SUM", "Lcals_GEN_LIN_RECUR",
                   "Lcals_HYDRO_1D", "Lcals_PLANCKIAN", "Lcals_TRIDIAG_ELIM",
                   "Polybench_JACOBI_1D", "Stream_ADD", "Stream_COPY",
                   "Stream_MUL", "Stream_TRIAD",

                   "Basic_IF_QUAD", "Basic_INDEXLIST", "Basic_INDEXLIST_3LOOP",
                   "Basic_NESTED_INIT", "Lcals_HYDRO_2D", "Lcals_INT_PREDICT",
                   "Polybench_FDTD_2D", "Polybench_HEAT_3D",
                   "Polybench_JACOBI_2D", "Stream_DOT", "Apps_CONVECTION3DPA",
                   "Apps_DEL_DOT_VEC_2D", "Apps_DIFFUSION3DPA", "Apps_ENERGY",
                   "Apps_FIR", "Apps_MASS3DPA", "Apps_NODAL_ACCUMULATION_3D",
                   "Apps_PRESSURE", "Apps_VOL3D", "Algorithm_SCAN",
                   "Algorithm_REDUCE_SUM", "Algorithm_MEMCPY", "Algorithm_MEMSET"],
   },
   "flops": {
      "kind": "GFLOPS",
      "kernels": [ "Basic_MAT_MAT_SHARED", "Polybench_2MM", "Polybench_3MM",
                   "Polybench_GEMM",

                   "Polybench_HEAT_3D", "Apps_CONVECTION3DPA",
                   "Apps_DEL_DOT_VEC_2D", "Apps_DIFFUSION3DPA", "Apps_FIR",
                   "Apps_MASS3DPA", "Apps_VOL3D", ],
   },
   "reduce": {
      "kind": "throughput(GProblem size/s)",
      "kernels": [ "Basic_PI_REDUCE", "Basic_REDUCE3_INT", "Basic_REDUCE_STRUCT",
                   "Basic_TRAP_INT", "Lcals_FIRST_MIN", "Stream_DOT",
                   "Algorithm_REDUCE_SUM",
                   
                   "Basic_PI_ATOMIC",]
   },
   "other": {
      "kind": "throughput(GProblem size/s)",
      "kernels": [ "Basic_PI_ATOMIC", "Polybench_ADI", "Polybench_ATAX",
                   "Polybench_FLOYD_WARSHALL", "Polybench_GEMVER",
                   "Polybench_GESUMMV", "Polybench_MVT",
                   "Apps_LTIMES", "Apps_LTIMES_NOVIEW", "Algorithm_SORT",
                   "Algorithm_SORTPAIRS", ]
   },
   "launch_bound": {
      "kind": "time/rep(us)",
      "kernels": [ "Apps_HALOEXCHANGE", "Apps_HALOEXCHANGE_FUSED", ]
   },
   }



g_color_base_factor = 1.0
g_color_lambda_factor = 0.7
g_color_raja_factor = 0.4
g_color_seq = normalize_color_tuple(make_color_tuple_rgb(204, 119, 34))  # ocre
g_color_omp = normalize_color_tuple(make_color_tuple_rgb(0, 115, 125))  # omp teal
g_color_ompt = normalize_color_tuple(make_color_tuple_rgb(125, 10, 0))  # omp teal compliment
g_color_cuda = normalize_color_tuple(make_color_tuple_rgb(118, 185, 0))  # nvidia green
g_color_hip = normalize_color_tuple(make_color_tuple_rgb(237, 28, 36))  # amd red
g_known_variants = {"Base_Seq"           : {"color": color_mul(g_color_seq, g_color_base_factor)},
                    "Lambda_Seq"         : {"color": color_mul(g_color_seq, g_color_lambda_factor)},
                    "RAJA_Seq"           : {"color": color_mul(g_color_seq, g_color_raja_factor)},

                    "Base_OpenMP"        : {"color": color_mul(g_color_omp, g_color_base_factor)},
                    "Lambda_OpenMP"      : {"color": color_mul(g_color_omp, g_color_lambda_factor)},
                    "RAJA_OpenMP"        : {"color": color_mul(g_color_omp, g_color_raja_factor)},

                    "Base_OpenMPTarget"  : {"color": color_mul(g_color_ompt, g_color_base_factor)},
                    "Lambda_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_lambda_factor)},
                    "RAJA_OpenMPTarget"  : {"color": color_mul(g_color_ompt, g_color_raja_factor)},

                    "Base_CUDA"          : {"color": color_mul(g_color_cuda, g_color_base_factor)},
                    "Lambda_CUDA"        : {"color": color_mul(g_color_cuda, g_color_lambda_factor)},
                    "RAJA_CUDA"          : {"color": color_mul(g_color_cuda, g_color_raja_factor)},

                    "Base_HIP"           : {"color": color_mul(g_color_hip, g_color_base_factor)},
                    "Lambda_HIP"         : {"color": color_mul(g_color_hip, g_color_lambda_factor)},
                    "RAJA_HIP"           : {"color": color_mul(g_color_hip, g_color_raja_factor)}
                    }
g_known_tunings = {"default"   : {"format": "-"},
                   "block_25"  : {"format": "-"},
                   "block_32"  : {"format": ":"},
                   "block_64"  : {"format": "-."},
                   "block_128" : {"format": "--"},
                   "block_256" : {"format": "-"},
                   "block_512" : {"format": "-."},
                   "block_1024": {"format": "-"},
                   "cub"       : {"format": ":"},
                   "library"   : {"format": "-"},
                   "rocprim"   : {"format": ":"}
                   }
g_markers = ["o", "s", "+", "x", "*", "d", "h", "p", "8"]

# reformat or color series
# formatted as series_name: dictionary of "color": color, "format": format
g_series_reformat = {}



def first(vals):
   return vals[0]


def last(vals):
   return vals[len(vals) - 1]


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
      stddev_val += (val - avg_val) * (val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val


def relstddev(vals):
   avg_val = avg(vals)
   stddev_val = 0
   for val in vals:
      stddev_val += (val - avg_val) * (val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val / abs(avg_val)


# returns (intercept, slope, correlation_coefficient)
def linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum):
   assert (n > 0)
   if n == 1:
      slope = 0.0
      intercept = ysum
      correlation_coefficient = 1.0
   else:
      slope = (n * xysum - xsum * ysum) / ((n * x2sum - xsum * xsum) + 1e-80)
      intercept = (ysum - slope * xsum) / n
      correlation_coefficient = (n * xysum - xsum * ysum) / (
               math.sqrt((n * x2sum - xsum * xsum) * (n * y2sum - ysum * ysum)) + 1e-80)
   return (intercept, slope, correlation_coefficient)


# returns (intercept, slope, correlation_coefficient)
def linearRegression(yvals, xvals):
   assert (len(xvals) == len(yvals))
   n = len(xvals)
   xsum = sum(xvals)
   ysum = sum(yvals)
   x2sum = sum([x * x for x in xvals])
   y2sum = sum([y * y for y in yvals])
   xysum = sum([xvals[i] * yvals[i] for i in range(0, n)])
   return linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum)


def eval_linearRegression(lr_vals, xval):
   return lr_vals[0] + lr_vals[1] * xval


# returns (intercept, slope, correlation_coefficient)
def linearRegression_loglog(yvals, xvals):
   assert (len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return linearRegression(ylogvals, xlogvals)


def eval_linearRegression_loglog(lr_vals, xval):
   return math.pow(2, lr_vals[0]) * math.pow(xval, lr_vals[1])


# returns (intercept, slope, correlation_coefficient)
def segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR):
   sums[0] += xvals[i]
   sums[1] += yvals[i]
   sums[2] += xvals[i] * xvals[i]
   sums[3] += yvals[i] * yvals[i]
   sums[4] += xvals[i] * yvals[i]
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
   break_point = None
   
   if i == 0:
      lr_vals_right = LR_right[i]
      break_point = xvals[i] - 1.0
   elif i > 0 and i < n:
      lr_vals_left = LR_left[i - 1]
      lr_vals_right = LR_right[i]
      break_point = (xvals[i - 1] + xvals[i]) / 2.0
   elif i == n:
      lr_vals_left = LR_left[i - 1]
      break_point = xvals[i - 1] + 1.0
   else:
      assert (0)
   
   numer = 0.0
   for j in range(0, n):
      xval = xvals[j]
      yval = yvals[j]
      lr_vals = None
      if xval < break_point:
         lr_vals = lr_vals_left
      else:
         lr_vals = lr_vals_right
      lr_yval = eval_linearRegression(lr_vals, xval)
      numer += (yval - lr_yval) * (yval - lr_yval)
   
   correlation_coefficient = 1.0 - numer / denom
   if correlation_coefficient > ret[2]:
      ret[0] = [break_point, ]
      ret[1] = [lr_vals_left, lr_vals_right, ]
      ret[2] = correlation_coefficient


# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression(yvals, xvals):
   assert (len(xvals) == len(yvals))
   N = len(xvals)
   
   LR_left = []
   LR_right = []
   for i in range(0, N):
      LR_left.append(None)
      LR_right.append(None)
   
   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for ii in range(0, N):
      i = N - ii - 1
      n = ii + 1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_right)
   
   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for i in range(0, N):
      n = i + 1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_left)
   
   yavg = avg(yvals)
   denom = sum([(y - yavg) * (y - yavg) for y in yvals])
   ret = [[], [], -math.inf]
   for i in range(0, N + 1):
      segmented_linearRegression_helper(ret, i, N, xvals, yvals, denom, LR_left, LR_right)
   
   return (*ret,)


def find_segment(break_points, xval):
   break_i = len(break_points)
   for i in range(0, len(break_points)):
      break_point = break_points[i]
      if xval < break_point:
         break_i = i
         break
   return break_i


def eval_segmented_linearRegression(slr_vals, xval):
   break_i = find_segment(slr_vals[0], xval)
   return eval_linearRegression(slr_vals[1][break_i], xval)


# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression_loglog(yvals, xvals):
   assert (len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return segmented_linearRegression(ylogvals, xlogvals)


def eval_segmented_linearRegression_loglog(slr_vals, xval):
   break_i = find_segment(slr_vals[0], math.log(xval, 2))
   return eval_linearRegression_loglog(slr_vals[1][break_i], xval)


class Data:
   num_sweeps = 0
   sweeps = {}
   sweep_markers = {}
   exclude_sweeps = {}
   
   num_run_sizes = 0
   run_sizes = {}
   
   include_kernel_groups = {}
   exclude_kernel_groups = {}
   
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
   
   def add_sweep(sweep_name):
      sweep_index = Data.num_sweeps
      Data.num_sweeps += 1
      Data.sweeps[sweep_name] = sweep_index
      Data.sweeps[sweep_index] = sweep_name
   
   def add_run_size(run_size_name):
      run_size_index = Data.num_run_sizes
      Data.num_run_sizes += 1
      Data.run_sizes[run_size_name] = run_size_index
      Data.run_sizes[run_size_index] = run_size_name
   
   def add_kernel(kernel_name):
      kernel_index = Data.num_kernels
      Data.num_kernels += 1
      Data.kernels[kernel_name] = kernel_index
      Data.kernels[kernel_index] = kernel_name
   
   def add_variant(variant_name):
      variant_index = Data.num_variants
      Data.num_variants += 1
      Data.variants[variant_name] = variant_index
      Data.variants[variant_index] = variant_name
      if variant_name in g_known_variants:
         variant_color = g_known_variants[variant_name]["color"]
         Data.variant_colors[variant_name] = variant_color
         Data.variant_colors[variant_index] = variant_color
      else:
         print("Unknown variant {0}".format(variant_name))
         sys.exit(1)
   
   def add_tuning(tuning_name):
      tuning_index = Data.num_tunings
      Data.num_tunings += 1
      Data.tunings[tuning_name] = tuning_index
      Data.tunings[tuning_index] = tuning_name
      if tuning_name in g_known_tunings:
         tuning_format = g_known_tunings[tuning_name]["format"]
         Data.tuning_formats[tuning_name] = tuning_format
         Data.tuning_formats[tuning_index] = tuning_format
      else:
         print("Unknown tuning {0}".format(tuning_name))
         sys.exit(1)
   
   num_axes = 5
   axes = {"sweep_dir_name": 0, 0: "sweep_dir_name",
           "run_size"      : 1, 1: "run_size",
           "kernel_index"  : 2, 2: "kernel_index",
           "variant_index" : 3, 3: "variant_index",
           "tuning_index"  : 4, 4: "tuning_index", }
   
   def get_axis_name(axis_index):
      if axis_index in Data.axes:
         return Data.axes[axis_index]
      else:
         raise NameError("Unknown axis index {}".format(axis_index))
   
   def get_index_name(axis_index, index):
      if axis_index == Data.axes["sweep_dir_name"]:
         return Data.sweeps[index]
      elif axis_index == Data.axes["run_size"]:
         return Data.run_sizes[index]
      elif axis_index == Data.axes["kernel_index"]:
         return Data.kernels[index]
      elif axis_index == Data.axes["variant_index"]:
         return Data.variants[index]
      elif axis_index == Data.axes["tuning_index"]:
         return Data.tunings[index]
      else:
         raise NameError("Unknown axis index {}".format(axis_index))
   
   def get_axis_index_str(axis_index, index):
      return "{}:{}".format(Data.get_axis_name(axis_index), Data.get_index_name(axis_index, index))
   
   def get_axes_index_str(axes_index):
      name = "{"
      for axis_index, index in axes_index.items():
         name = "{}{},".format(name, Data.get_axis_index_str(axis_index, index))
      return "{}}}".format(name)
   
   def get_axis_index(axis_name, index_name):
      if axis_name == "sweep_dir_name":
         return {Data.axes[axis_name]: Data.sweeps[index_name], }
      elif axis_name == "run_size":
         return {Data.axes[axis_name]: Data.run_sizes[index_name], }
      elif axis_name == "kernel_index":
         return {Data.axes[axis_name]: Data.kernels[index_name], }
      elif axis_name == "variant_index":
         return {Data.axes[axis_name]: Data.variants[index_name], }
      elif axis_name == "tuning_index":
         return {Data.axes[axis_name]: Data.tunings[index_name], }
      else:
         raise NameError("Unknown axis name {}".format(axis_name))
   
   def axes_difference(axes, partial_axes_index):
      new_axes = []
      for axis_index in axes:
         if not axis_index in partial_axes_index:
            new_axes.append(axis_index)
      return new_axes
   
   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   info_axes = [axes["sweep_dir_name"],
                axes["run_size"],
                axes["kernel_index"], ]
   
   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   #           variant index - for variants
   #             tuning index - for tunings
   data_axes = [axes["sweep_dir_name"],
                axes["run_size"],
                axes["kernel_index"],
                axes["variant_index"],
                axes["tuning_index"], ]
   
   # multi-dimensional array structured like data but missing some dimensions
   #     directory name - platform, compiler, etc
   #       kernel index - for kernels
   #         variant index - for variants
   #           tuning index - for tunings
   run_size_reduced_axes = [axes["sweep_dir_name"],
                            axes["kernel_index"],
                            axes["variant_index"],
                            axes["tuning_index"], ]
   
   data_model_kind = "time(s)"
   
   def MultiAxesTreeKeyGenerator0(data_tree):
      assert (len(data_tree.axes) == 0)
      if False:
         yield {} #unreachable
   
   def MultiAxesTreeKeyGenerator1(data_tree):
      assert (len(data_tree.axes) == 1)
      assert (data_tree.data)
      for k0 in data_tree.data.keys():
         yield {data_tree.axes[0]: k0, }
   
   def MultiAxesTreeKeyGenerator2(data_tree):
      assert (len(data_tree.axes) == 2)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1 in v0.keys():
            yield {data_tree.axes[0]: k0,
                   data_tree.axes[1]: k1, }
   
   def MultiAxesTreeKeyGenerator3(data_tree):
      assert (len(data_tree.axes) == 3)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2 in v1.keys():
               yield {data_tree.axes[0]: k0,
                      data_tree.axes[1]: k1,
                      data_tree.axes[2]: k2, }
   
   def MultiAxesTreeKeyGenerator4(data_tree):
      assert (len(data_tree.axes) == 4)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3 in v2.keys():
                  yield {data_tree.axes[0]: k0,
                         data_tree.axes[1]: k1,
                         data_tree.axes[2]: k2,
                         data_tree.axes[3]: k3, }
   
   def MultiAxesTreeKeyGenerator5(data_tree):
      assert (len(data_tree.axes) == 5)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  for k4 in v3.keys():
                     yield {data_tree.axes[0]: k0,
                            data_tree.axes[1]: k1,
                            data_tree.axes[2]: k2,
                            data_tree.axes[3]: k3,
                            data_tree.axes[4]: k4, }
   
   def MultiAxesTreeItemGenerator0(data_tree):
      assert (len(data_tree.axes) == 0)
      if False:
         yield ({}, None,) #unreachable
   
   def MultiAxesTreeItemGenerator1(data_tree):
      assert (len(data_tree.axes) == 1)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         yield ({data_tree.axes[0]: k0, }, v0,)
   
   def MultiAxesTreeItemGenerator2(data_tree):
      assert (len(data_tree.axes) == 2)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            yield ({data_tree.axes[0]: k0,
                    data_tree.axes[1]: k1, }, v1,)
   
   def MultiAxesTreeItemGenerator3(data_tree):
      assert (len(data_tree.axes) == 3)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               yield ({data_tree.axes[0]: k0,
                       data_tree.axes[1]: k1,
                       data_tree.axes[2]: k2, }, v2,)
   
   def MultiAxesTreeItemGenerator4(data_tree):
      assert (len(data_tree.axes) == 4)
      assert (data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  yield ({data_tree.axes[0]: k0,
                          data_tree.axes[1]: k1,
                          data_tree.axes[2]: k2,
                          data_tree.axes[3]: k3, }, v3,)
   
   def MultiAxesTreeItemGenerator5(data_tree):
      assert (len(data_tree.axes) == 5)
      assert (data_tree.data)
      
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  for k4, v4 in v3.items():
                     yield ({data_tree.axes[0]: k0,
                             data_tree.axes[1]: k1,
                             data_tree.axes[2]: k2,
                             data_tree.axes[3]: k3,
                             data_tree.axes[4]: k4, }, v4,)
   
   def MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                                                axes_index, leftover_axes_index,
                                                val, depth):
      if data_tree.axes[depth] in partial_axes_index:
         key = partial_axes_index[data_tree.axes[depth]]
         if key in val:
            val = val[key]
            axes_index[data_tree.axes[depth]] = key
            if depth + 1 == len(data_tree.axes):
               yield (axes_index.copy(), leftover_axes_index.copy(), val,)
            else:
               gen = Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                                                                   axes_index, leftover_axes_index, val, depth + 1)
               for yld in gen:
                  yield yld
         else:
            # print(data_tree, partial_axes_index,
            #      axes_index, leftover_axes_index,
            #      key, val, depth)
            raise NameError("invalid index {} {}".format(Data.get_axes_index_str(axes_index),
                                                         Data.get_axis_index_str(data_tree.axes[depth], key)))
      else:
         for key, val in val.items():
            axes_index[data_tree.axes[depth]] = key
            leftover_axes_index[data_tree.axes[depth]] = key
            if depth + 1 == len(data_tree.axes):
               yield (axes_index.copy(), leftover_axes_index.copy(), val,)
            else:
               gen = Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                                                                   axes_index, leftover_axes_index, val, depth + 1)
               for yld in gen:
                  yield yld
   
   def MultiAxesTreePartialItemGenerator0(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 0)
      if False:
         yield ({}, None,) #unreachable
   
   def MultiAxesTreePartialItemGenerator1(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 1)
      assert (data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)
   
   def MultiAxesTreePartialItemGenerator2(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 2)
      assert (data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)
   
   def MultiAxesTreePartialItemGenerator3(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 3)
      assert (data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)
   
   def MultiAxesTreePartialItemGenerator4(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 4)
      assert (data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)
   
   def MultiAxesTreePartialItemGenerator5(data_tree, partial_axes_index):
      assert (len(data_tree.axes) == 5)
      assert (data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)
   
   class MultiAxesTree:
      # axes is an array of axis_indices in the depth order they occur in the tree
      # indices is a dictionary of axis_indices to indices
      
      def __init__(self, axes):
         assert (axes)
         self.axes = axes
         self.data = {}
      
      def check(self, axes_index):
         data = self.data
         for axis_index in self.axes:
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(Data.get_axis_name(axis_index)))
            index = axes_index[axis_index]
            if not index in data:
               return False
            data = data[index]
         return True
      
      def get(self, axes_index):
         data = self.data
         for axis_index in self.axes:
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(Data.get_axis_name(axis_index)))
            index = axes_index[axis_index]
            if not index in data:
               raise NameError("Missing index {}".format(Data.get_axes_index_str(axes_index,index)))
            data = data[index]
         return data
      
      def set(self, axes_index, val):
         data = self.data
         for i in range(0, len(self.axes) - 1):
            axis_index = self.axes[i]
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(Data.get_axis_name(axis_index)))
            index = axes_index[axis_index]
            if not index in data:
               data[index] = {}
            data = data[index]
         axis_index = self.axes[len(self.axes) - 1]
         if not axis_index in axes_index:
            axis_name = Data.axes[axis_index]
            raise NameError("Missing axis {}".format(Data.get_axis_name(axis_index)))
         index = axes_index[axis_index]
         data[index] = val
      
      def indexName(self, axes_index):
         name = ""
         for axis_index, index in axes_index.items():
            if name:
               name = "{} {}".format(name, Data.get_index_name(axis_index, index))
            else:
               name = Data.get_index_name(axis_index, index)
         return name
      
      def axesString(self):
         axes_names = ""
         for axis_index in self.axes:
            if axes_names:
               axes_names = "{}, {}".format(axes_names, Data.get_axis_name(axis_index))
            else:
               axes_names = "[{}".format(Data.axes[axis_index])
         return "{}]".format(axes_names)
      
      def dataString(self, compact=True):
         if compact:
            buf = self.axesString() + "\n"
         else:
            buf = ""
         for item in self.items():
            keys, value = item
            #print(str(keys))
            if compact:
               buf += str(item) + "\n"
            else:
               buf += "(" + Data.get_axes_index_str(keys) + "," + str(value) + ")\n"
         return buf
      
      def __repr__(self):
         return "MultiAxesTree({}):\n{}".format(self.axesString(), self.dataString())
      
      def __str__(self):
         return "MultiAxesTree({})".format(self.axesString())
      
      def keys(self):
         assert (self.data != None)
         if len(self.axes) == 0:
            return Data.MultiAxesTreeKeyGenerator0(self)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreeKeyGenerator1(self)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreeKeyGenerator2(self)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreeKeyGenerator3(self)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreeKeyGenerator4(self)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreeKeyGenerator5(self)
         else:
            raise ValueError
      
      def items(self):
         assert (self.data != None)
         if len(self.axes) == 0:
            return Data.MultiAxesTreeItemGenerator0(self)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreeItemGenerator1(self)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreeItemGenerator2(self)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreeItemGenerator3(self)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreeItemGenerator4(self)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreeItemGenerator5(self)
         else:
            raise ValueError
      
      def partial_match_items(self, partial_axes_index):
         assert (self.data != None)
         num_matching_indices = 0
         for axis_index in self.axes:
            if axis_index in partial_axes_index:
               num_matching_indices += 1
         assert (num_matching_indices == len(partial_axes_index))
         if len(self.axes) == 0:
            return Data.MultiAxesTreePartialItemGenerator0(self, partial_axes_index)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreePartialItemGenerator1(self, partial_axes_index)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreePartialItemGenerator2(self, partial_axes_index)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreePartialItemGenerator3(self, partial_axes_index)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreePartialItemGenerator4(self, partial_axes_index)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreePartialItemGenerator5(self, partial_axes_index)
         else:
            raise ValueError
      
      def __iter__(self):
         return self.keys()
   
   class DataTree:
      
      def __init__(self, kind, label, model_kind=None, axes=None, args=None, func=None):
         #print("DataTree init:"+str(kind)+ ' ' + str(label) + ' ' + str(args))
         self.kind = kind
         self.label = label
         self.axes = axes
         self.args = args
         self.func = func
         self.model_kind = model_kind
         if not self.model_kind and self.args:
            self.model_kind = self.args[0]
         self.data = None
      
      def makeData(self, axes=None):
         if not self.axes:
            if axes:
               self.axes = axes
            elif self.model_kind and self.model_kind in Data.kinds:
               self.axes = Data.kinds[self.model_kind].axes
         assert (self.axes)
         self.data = Data.MultiAxesTree(self.axes)
      
      def hasAxes(self, other_axes):
         for axis_index in other_axes:
            if not axis_index in self.axes:
               return False
         return True
      
      def sameAxes(self, other_axes):
         if len(self.axes) != len(other_axes):
            return False
         self.hasAxes(other_axes)
      
      def missingAxes(self, other_axes):
         for axis_index in other_axes:
            if not axis_index in self.axes:
               return True
         return False
      
      def check(self, axes_index):
         return self.data.check(axes_index)
      
      def get(self, axes_index):
         return self.data.get(axes_index)
      
      def set(self, axes_index, val):
         return self.data.set(axes_index, val)
      
      def keys(self):
         return self.data.keys()
      
      def items(self):
         return self.data.items()
      
      def partial_match_items(self, partial_axes_index):
         return self.data.partial_match_items(partial_axes_index)
      
      def __iter__(self):
         return iter(self.data)
      
      def indexName(self, axes_index):
         return self.data.indexName(axes_index)
      
      def axesString(self):
         return self.data.axesString()
      
      def dataString(self, compact=True):
         return self.data.dataString(compact)
      
      def __repr__(self):
         return "DataTree({} {} {}):\n{}".format(self.kind, self.label, self.axesString(), self.dataString())
      
      def __str__(self):
         return "DataTree({} {} {})".format(self.kind, self.label, self.axesString())
   
   class DataTreeTemplate:
      
      def __init__(self, kind_template, label_template,
                   combined_axis=None, model_kind=None, args=None, func=None):
         self.kind_template = kind_template
         self.label_template = label_template
         self.combined_axis_template = combined_axis
         self.model_kind_template = model_kind
         self.arg_templates = args
         self.func = func
      
      def getKind(self, template_args):
         return self.kind_template.format(*template_args)
      
      def getLabel(self, template_args):
         arg_labels = [arg_kind in Data.kinds and Data.kinds[arg_kind].label or None for arg_kind in template_args]
         return self.label_template.format(*arg_labels)
      
      def getArgs(self, template_args):
         return [arg.format(*template_args) for arg in self.arg_templates]
      
      def getCombinedAxis(self, template_args):
         return self.combined_axis_template.format(*template_args)
      
      def getModelKind(self, args, template_args):
         assert (len(args) > 0)
         model_kind = None
         # choose model_kind with most axes
         for kind in args:
            if kind in Data.kinds:
               if not model_kind:
                  model_kind = kind
               elif len(Data.kinds[kind].axes) > len(Data.kinds[model_kind].axes):
                  model_kind = kind
         # use chosen model_kind
         if self.model_kind_template:
            model_kind = self.model_kind_template.format(*template_args)
         assert (model_kind)
         return model_kind
      
      def getAxes(self, model_kind, template_args):
         model_axes = Data.kinds[model_kind].axes
         combined_axis_index = None
         if self.combined_axis_template:
            combined_axis_name = self.getCombinedAxis(template_args)
            combined_axis_index = Data.axes[combined_axis_name]
         axes = []
         for axis_index in model_axes:
            if axis_index != combined_axis_index:
               axes.append(axis_index)
         return axes
      
      def makeDataTree(self, template_args):
         kind = self.getKind(template_args)
         label = self.getLabel(template_args)
         args = self.getArgs(template_args)
         model_kind = self.getModelKind(args, template_args)
         axes = self.getAxes(model_kind, template_args)
         return Data.DataTree(kind, label, model_kind=model_kind, axes=axes, args=args, func=self.func)


# has info derivable from first kind "time(s)" which is read from files
   kinds = {"Problem size"               : DataTree("Problem size", "Problem size", axes=info_axes),
            "Reps"                       : DataTree("Reps", "Reps", axes=info_axes),
            "Iterations/rep"             : DataTree("Iterations/rep", "Iterations", axes=info_axes),
            "Kernels/rep"                : DataTree("Kernels/rep", "Kernels", axes=info_axes),
            "Bytes/rep"                  : DataTree("Bytes/rep", "Bytes", axes=info_axes),
            "FLOPS/rep"                  : DataTree("FLOPS/rep", "FLOPS", axes=info_axes),
   
            "time(s)"                    : DataTree("time(s)", "time(s)", axes=data_axes),
   
            "time(ms)"                   : DataTree("time(ms)", "time(ms)", args=["time(s)"], func=lambda t: t * 1000.0),
            "time(us)"                   : DataTree("time(us)", "time(us)", args=["time(s)"],
                                                    func=lambda t: t * 1000000.0),
            "time(ns)"                   : DataTree("time(ns)", "time(ns)", args=["time(s)"],
                                                    func=lambda t: t * 1000000000.0),
   
            "time/rep(s)"                : DataTree("time/rep(s)", "time(s)", args=["time(s)", "Reps"],
                                                    func=lambda t, r: t / r),
            "time/rep(ms)"               : DataTree("time/rep(ms)", "time(ms)", args=["time/rep(s)"],
                                                    func=lambda tpr: tpr * 1000.0),
            "time/rep(us)"               : DataTree("time/rep(us)", "time(us)", args=["time/rep(s)"],
                                                    func=lambda tpr: tpr * 1000000.0),
            "time/rep(ns)"               : DataTree("time/rep(ns)", "time(ns)", args=["time/rep(s)"],
                                                    func=lambda tpr: tpr * 1000000000.0),
   
            "time/it(s)"                 : DataTree("time/it(s)", "time(s)", args=["time/rep(s)", "Iterations/rep"],
                                                    func=lambda tpr, ipr: tpr / ipr),
            "time/it(ms)"                : DataTree("time/it(ms)", "time(ms)", args=["time/it(s)"],
                                                    func=lambda tpi: tpi * 1000.0),
            "time/it(us)"                : DataTree("time/it(us)", "time(us)", args=["time/it(s)"],
                                                    func=lambda tpi: tpi * 1000000.0),
            "time/it(ns)"                : DataTree("time/it(ns)", "time(ns)", args=["time/it(s)"],
                                                    func=lambda tpi: tpi * 1000000000.0),
   
            "time/kernel(s)"             : DataTree("time/kernel(s)", "time(s)", args=["time/rep(s)", "Kernels/rep"],
                                                    func=lambda tpr, kpr: tpr / kpr),
            "time/kernel(ms)"            : DataTree("time/kernel(ms)", "time(ms)", args=["time/kernel(s)"],
                                                    func=lambda tpk: tpk * 1000.0),
            "time/kernel(us)"            : DataTree("time/kernel(us)", "time(us)", args=["time/kernel(s)"],
                                                    func=lambda tpk: tpk * 1000000.0),
            "time/kernel(ns)"            : DataTree("time/kernel(ns)", "time(ns)", args=["time/kernel(s)"],
                                                    func=lambda tpk: tpk * 1000000000.0),
   
            "throughput(Problem size/s)" : DataTree("throughput(Problem size/s)", "throughput(Problem size/s)",
                                                    args=["time/rep(s)", "Problem size"], func=lambda tpr, ps: ps / tpr),
            "throughput(Problem size/ms)": DataTree("throughput(Problem size/ms)", "throughput(Problem size/ms)",
                                                    args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000.0),
            "throughput(Problem size/us)": DataTree("throughput(Problem size/us)", "throughput(Problem size/us)",
                                                    args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000.0),
            "throughput(Problem size/ns)": DataTree("throughput(Problem size/ns)", "throughput(Problem size/ns)",
                                                    args=["throughput(Problem size/s)"],
                                                    func=lambda thr: thr / 1000000000.0),
            "throughput(KProblem size/s)": DataTree("throughput(KProblem size/s)", "throughput(KProblem size/s)",
                                                    args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000.0),
            "throughput(MProblem size/s)": DataTree("throughput(MProblem size/s)", "throughput(MProblem size/s)",
                                                    args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000.0),
            "throughput(GProblem size/s)": DataTree("throughput(GProblem size/s)", "throughput(GProblem size/s)",
                                                    args=["throughput(Problem size/s)"],
                                                    func=lambda thr: thr / 1000000000.0),
            "throughput(TProblem size/s)": DataTree("throughput(TProblem size/s)", "throughput(TProblem size/s)",
                                                    args=["throughput(Problem size/s)"],
                                                    func=lambda thr: thr / 1000000000000.0),
   
            "bandwidth(B/s)"             : DataTree("bandwidth(B/s)", "bandwidth(B/s)", args=["time/rep(s)", "Bytes/rep"],
                                                    func=lambda tpr, bpr: bpr / tpr),
            "bandwidth(KB/s)"            : DataTree("bandwidth(KB/s)", "bandwidth(KB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1000.0),
            "bandwidth(MB/s)"            : DataTree("bandwidth(MB/s)", "bandwidth(MB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1000000.0),
            "bandwidth(GB/s)"            : DataTree("bandwidth(GB/s)", "bandwidth(GB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1000000000.0),
            "bandwidth(TB/s)"            : DataTree("bandwidth(TB/s)", "bandwidth(TB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1000000000000.0),
            "bandwidth(KiB/s)"           : DataTree("bandwidth(KiB/s)", "bandwidth(KiB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1024.0),
            "bandwidth(MiB/s)"           : DataTree("bandwidth(MiB/s)", "bandwidth(MiB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1048576.0),
            "bandwidth(GiB/s)"           : DataTree("bandwidth(GiB/s)", "bandwidth(GiB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1073741824.0),
            "bandwidth(TiB/s)"           : DataTree("bandwidth(TiB/s)", "bandwidth(TiB/s)", args=["bandwidth(B/s)"],
                                                    func=lambda bps: bps / 1099511627776.0),
   
            "FLOPS"                      : DataTree("FLOPS", "FLOPS", args=["time/rep(s)", "FLOPS/rep"],
                                                    func=lambda tpr, fpr: fpr / tpr),
            "KFLOPS"                     : DataTree("KFLOPS", "KFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000.0),
            "MFLOPS"                     : DataTree("MFLOPS", "MFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000000.0),
            "GFLOPS"                     : DataTree("GFLOPS", "GFLOPS", args=["FLOPS"],
                                                    func=lambda fps: fps / 1000000000.0),
            "TFLOPS"                     : DataTree("TFLOPS", "TFLOPS", args=["FLOPS"],
                                                    func=lambda fps: fps / 1000000000000.0),
      
            }
   
   kind_templates = {
      "log10"                         : DataTreeTemplate("log10<{0}>", "log10({0})", args=["{0}", ],
                                                         func=lambda val: math.log(val, 10)),
      "log2"                          : DataTreeTemplate("log2<{0}>", "log2({0})", args=["{0}", ],
                                                         func=lambda val: math.log(val, 2)),
      "ln"                            : DataTreeTemplate("ln<{0}>", "ln({0})", args=["{0}", ],
                                                         func=lambda val: math.log(val)),
      
      "add"                           : DataTreeTemplate("add<{0},{1}>", "{0} + {1}", args=["{0}", "{1}"],
                                                         func=lambda lhs, rhs: lhs + rhs),
      "sub"                           : DataTreeTemplate("sub<{0},{1}>", "{0} - {1}", args=["{0}", "{1}"],
                                                         func=lambda lhs, rhs: lhs - rhs),
      "mul"                           : DataTreeTemplate("mul<{0},{1}>", "{0} * {1}", args=["{0}", "{1}"],
                                                         func=lambda lhs, rhs: lhs * rhs),
      "div"                           : DataTreeTemplate("div<{0},{1}>", "{0} / {1}", args=["{0}", "{1}"],
                                                         func=lambda lhs, rhs: lhs / rhs),
      
      "first"                         : DataTreeTemplate("first<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=first),
      "last"                          : DataTreeTemplate("last<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=last),
      "min"                           : DataTreeTemplate("min<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=min),
      "max"                           : DataTreeTemplate("max<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=max),
      "sum"                           : DataTreeTemplate("sum<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=sum),
      "avg"                           : DataTreeTemplate("avg<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=avg),
      "stddev"                        : DataTreeTemplate("stddev<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=stddev),
      "relstddev"                     : DataTreeTemplate("relstddev<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"],
                                                         func=relstddev),
      
      "_LR"                           : DataTreeTemplate("_LR<{0}>", "intercept, slope, correlation coefficient",
                                                         combined_axis="run_size", args=["{0}", "Problem size"],
                                                         func=linearRegression),
      "LR_intercept"                  : DataTreeTemplate("LR_intercept<{0}>", "intercept", args=["_LR<{0}>"],
                                                         func=lambda lr: lr[0]),
      "LR_slope"                      : DataTreeTemplate("LR_slope<{0}>", "slope", args=["_LR<{0}>"],
                                                         func=lambda lr: lr[1]),
      "LR_correlationCoefficient"     : DataTreeTemplate("LR_correlationCoefficient<{0}>", "correlation coefficient",
                                                         args=["_LR<{0}>"], func=lambda lr: lr[2]),
      "LR"                            : DataTreeTemplate("LR<{0}>", "{0}", model_kind="{0}",
                                                         args=["_LR<{0}>", "Problem size"], func=eval_linearRegression),
      
      "_LR_log"                       : DataTreeTemplate("_LR_log<{0}>", "intercept, slope, correlation coefficient",
                                                         combined_axis="run_size", args=["{0}", "Problem size"],
                                                         func=linearRegression_loglog),
      "LR_log_intercept"              : DataTreeTemplate("LR_log_intercept<{0}>", "intercept", args=["_LR_log<{0}>"],
                                                         func=lambda lr: lr[0]),
      "LR_log_slope"                  : DataTreeTemplate("LR_log_slope<{0}>", "slope", args=["_LR_log<{0}>"],
                                                         func=lambda lr: lr[1]),
      "LR_log_correlationCoefficient" : DataTreeTemplate("LR_log_correlationCoefficient<{0}>", "correlation coefficient",
                                                         args=["_LR_log<{0}>"], func=lambda lr: lr[2]),
      "LR_log"                        : DataTreeTemplate("LR_log<{0}>", "{0}", model_kind="{0}",
                                                         args=["_LR_log<{0}>", "Problem size"],
                                                         func=eval_linearRegression_loglog),
      
      "_LR2"                          : DataTreeTemplate("_LR2<{0}>", "intercept, slope, correlation coefficient",
                                                         combined_axis="run_size", args=["{0}", "Problem size"],
                                                         func=segmented_linearRegression),
      "LR2_intercept"                 : DataTreeTemplate("LR2_intercept<{0}>", "intercept", args=["_LR2<{0}>"],
                                                         func=lambda lr: lr[0]),
      "LR2_slope"                     : DataTreeTemplate("LR2_slope<{0}>", "slope", args=["_LR2<{0}>"],
                                                         func=lambda lr: lr[1]),
      "LR2_correlationCoefficient"    : DataTreeTemplate("LR2_correlationCoefficient<{0}>", "correlation coefficient",
                                                         args=["_LR2<{0}>"], func=lambda lr: lr[2]),
      "LR2"                           : DataTreeTemplate("LR2<{0}>", "{0}", model_kind="{0}",
                                                         args=["_LR2<{0}>", "Problem size"],
                                                         func=eval_segmented_linearRegression),
      
      "_LR2_log"                      : DataTreeTemplate("_LR2_log<{0}>", "intercept, slope, correlation coefficient",
                                                         combined_axis="run_size", args=["{0}", "Problem size"],
                                                         func=segmented_linearRegression_loglog),
      "LR2_log_intercept"             : DataTreeTemplate("LR2_log_intercept<{0}>", "intercept", args=["_LR2_log<{0}>"],
                                                         func=lambda lr: lr[0]),
      "LR2_log_slope"                 : DataTreeTemplate("LR2_log_slope<{0}>", "slope", args=["_LR2_log<{0}>"],
                                                         func=lambda lr: lr[1]),
      "LR2_log_correlationCoefficient": DataTreeTemplate("LR2_log_correlationCoefficient<{0}>", "correlation coefficient",
                                                         args=["_LR2_log<{0}>"], func=lambda lr: lr[2]),
      "LR2_log"                       : DataTreeTemplate("LR2_log<{0}>", "{0}", model_kind="{0}",
                                                         args=["_LR2_log<{0}>", "Problem size"],
                                                         func=eval_segmented_linearRegression_loglog),
      
   }

   def compute_data(kind):
      if not kind in Data.kinds:
         raise NameError("Unknown data kind {}".format(kind))
   
      datatree = Data.kinds[kind]
      if datatree.data:
         return  # already calculated
      if not (datatree.model_kind and datatree.args and datatree.func):
         raise NameError("Computing data is not supported for kind {0}".format(kind))
   
      model_kind = datatree.model_kind
      compute_args = datatree.args
      compute_func = datatree.func
   
      if model_kind != kind:
         Data.compute(model_kind)
   
      arg_datatrees = ()
      for arg_kind in compute_args:
         # calculate data for arg_kind
         Data.compute(arg_kind)
         arg_datatree = Data.kinds[arg_kind]
         arg_datatrees = arg_datatrees + (arg_datatree,)
   
      if (not model_kind in Data.kinds) or (not Data.kinds[model_kind].data):
         raise NameError("Model data not available {0}, no args".format(model_kind))
   
      datatree.makeData()
   
      use_lists = ()
      for arg_datatree in arg_datatrees:
         use_list = datatree.missingAxes(arg_datatree.axes)
         use_lists = use_lists + (use_list,)
   
      for axes_index in Data.kinds[model_kind]:
         # print("compute_data:"+str(axes_index))
         if not datatree.check(axes_index):
            args_val = ()
            for i in range(0, len(arg_datatrees)):
               arg_datatree = arg_datatrees[i]
               arg_val = arg_datatree.get(axes_index)
               if use_lists[i]:
                  arg_val = [arg_val, ]
               args_val = args_val + (arg_val,)
            datatree.set(axes_index, args_val)
         else:
            args_val = datatree.get(axes_index)
            for i in range(0, len(arg_datatrees)):
               if use_lists[i]:
                  arg_datatree = arg_datatrees[i]
                  arg_val = arg_datatree.get(axes_index)
                  args_val[i].append(arg_val)
   
      for axes_index, args_val in datatree.items():
         val = compute_func(*args_val)
         datatree.set(axes_index, val)

   def compute_index(kind_preindex, index_args):
      # print("compute_index", kind_preindex, index_args)
      Data.compute(kind_preindex)
      datatree_preindex = Data.kinds[kind_preindex]
   
      # extract axes and indices
      partial_axis_index = {}
      for index_str in index_args:
         index_list = index_str.split("::")
         if len(index_list) != 2:
            raise NameError("Expected valid index <axis>::<index>: {}".format(index_str))
         axis_name = index_list[0].strip()
         index_name = index_list[1].strip()
         partial_axis_index.update(Data.get_axis_index(axis_name, index_name))
   
      kind = "{}[{}]".format(kind_preindex, ",".join(index_args))
   
      datatree = None
      if kind in Data.kinds:
         datatree = Data.kinds[kind]
         if datatree.data:
            return
      else:
         axes = Data.axes_difference(datatree_preindex.axes, partial_axis_index)
         datatree = Data.DataTree(kind, datatree_preindex.label, axes=axes)
         Data.kinds[kind] = datatree
   
      datatree.makeData()
   
      for axes_index, partial_axes_index, value in datatree_preindex.partial_match_items(partial_axis_index):
         datatree.set(partial_axes_index, value)

   def compute_templated_data(kind_template, template_args):
      # print("compute_templated_data", kind_template, template_args)
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
      # print("kind_template_scan", kind)
   
      kind_prefix = None
   
      template_args = []
      index_args = []
   
      template_depth = 0
      index_depth = 0
   
      arg_end_idx = -1
   
      # look through string backwards to find indexing or templating
      for i_forward in range(0, len(kind)):
         i = len(kind) - i_forward - 1
         c = kind[i]
         if c == ">" or c == "]":
            if template_depth == 0 and index_depth == 0:
               arg_end_idx = i
            if c == ">":
               template_depth += 1
            elif c == "]":
               index_depth += 1
         elif c == ",":
            if template_depth == 1 and index_depth == 0:
               template_args.append(kind[i + 1:arg_end_idx].strip())
               arg_end_idx = i
            elif template_depth == 0 and index_depth == 1:
               index_args.append(kind[i + 1:arg_end_idx].strip())
               arg_end_idx = i
         elif c == "<" or c == "[":
            if template_depth == 1 and index_depth == 0:
               template_args.append(kind[i + 1:arg_end_idx].strip())
               arg_end_idx = -1
            elif template_depth == 0 and index_depth == 1:
               index_args.append(kind[i + 1:arg_end_idx].strip())
               arg_end_idx = -1
            if c == "<":
               template_depth -= 1
            elif c == "[":
               index_depth -= 1
            if template_depth == 0 and index_depth == 0:
               if not kind_prefix:
                  kind_prefix = kind[:i].strip()
                  break
      assert (arg_end_idx == -1)
      assert (template_depth == 0)
      assert (index_depth == 0)
      assert (kind_prefix)
   
      # reverse lists
      for i in range(0, len(template_args) // 2):
         i_rev = len(template_args) - i - 1
         template_args[i], template_args[i_rev] = template_args[i_rev], template_args[i]
      for i in range(0, len(index_args) // 2):
         i_rev = len(index_args) - i - 1
         index_args[i], index_args[i_rev] = index_args[i_rev], index_args[i]
   
      return (kind_prefix, template_args, index_args)

   def compute(kind):
      if kind in Data.kinds:
         if not Data.kinds[kind].data:
            Data.compute_data(kind)
         else:
            pass
      else:
         kind_template, template_args, index_args = Data.kind_template_scan(kind)
         # print("Data.kind_template_scan", kind_template, template_args, index_args)
         if template_args:
            if kind_template in Data.kind_templates:
               Data.compute_templated_data(kind_template, template_args)
            else:
               raise NameError("Unknown data kind template {}".format(kind))
         elif index_args:
            Data.compute_index(kind_template, index_args)
         else:
            raise NameError("Unknown data kind {}".format(kind))


