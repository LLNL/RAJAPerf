#!/usr/bin/env python3
import argparse
import sys
import os
import csv
import glob

def get_size_from_dir_name(sweep_subdir_name):
   # print(sweep_subdir_name)
   run_size_name = sweep_subdir_name.replace("SIZE_", "")
   try:
      run_size = int(run_size_name)
      return str(run_size)
   except ValueError:
      raise NameError("Expected SIZE_<run_size>".format(sweep_subdir_name))

def prescan_sweep_dirs(sweep_dir_paths) -> dict:
   prescan = {"kernels_union" : [], "kernels_intersection" : [], "variants" : [], "tunings" : [], "sweep_sizes": [], "machines": []}
   # machines only gleans os.path.basename of sweep_dir_paths, and does not actually parse real encoded machine names from data; so machine_name is a convention for directory naming
   sets = []
   for sweep_dir_path in sweep_dir_paths:
      kernel_set = set()
      sweep_dir_path = sweep_dir_path.rstrip(os.sep)
      sweep_dir_name = os.path.basename(sweep_dir_path)
      if sweep_dir_name not in prescan["machines"]:
         prescan["machines"].append(sweep_dir_name)
      subdirs = sorted(glob.glob(glob.escape(sweep_dir_path) + os.sep + "**" + os.sep + "SIZE_*",recursive=True))
      for subdir in subdirs:
         #print(subdir)
         run_size = get_size_from_dir_name(os.path.basename(subdir))
         if run_size not in prescan["sweep_sizes"]:
            prescan["sweep_sizes"].append(run_size)
         # open one of the timing files at this run_size
         timing_files = sorted(glob.glob(glob.escape(subdir) + os.sep + "RAJAPerf-timing-*",recursive=False))
         with open(timing_files[0],"r") as file:
            file_reader = csv.reader(file, delimiter=',')
            variants_read = False
            tunings_read = False
            for row in file_reader:
               #print(row)
               if row[0].strip() == "Kernel":
                  if not variants_read:
                     for c in range(1, len(row)):
                        variant_name = row[c].strip()
                        if variant_name not in prescan["variants"]:
                           prescan["variants"].append(variant_name)
                     variants_read = True
                  elif not tunings_read:
                     for c in range(1, len(row)):
                        tuning_name = row[c].strip()
                        if tuning_name not in prescan["tunings"]:
                           prescan["tunings"].append(tuning_name)
                     tunings_read = True
               elif variants_read == True & tunings_read == True:
                  kernel_name = row[0].strip()
                  if kernel_name not in prescan["kernels_union"]:
                     prescan["kernels_union"].append(kernel_name)
                  if kernel_name not in kernel_set:
                     kernel_set.add(kernel_name)
      sets.append(kernel_set)
   prescan["kernels_intersection"] = set.intersection(*sets)
   return prescan

class KernelAction(argparse.Action):
   
   def __init__(self, option_strings, dest, nargs='+', **kwargs):
      super().__init__(option_strings, dest, nargs,**kwargs)
   
   def __call__(self, parser, namespace, values, option_string=None):
      print('Kernel Action Namespace=%r values=%r option_string=%r' % (namespace, values, option_string))
      check_kernels = []
      for k in values:
         if k in namespace.candidate_kernels:
            check_kernels.append(k)
      setattr(namespace, self.dest, check_kernels)  # do the normal attr set for dest

class DirectoryAction(argparse.Action):
   def __init__(self, option_strings, dest, nargs='+', **kwargs):
      if nargs != '+':
         raise ValueError("Expected variable nargs to be set to '+'")
      super().__init__(option_strings, dest,nargs, **kwargs)
   
   def __call__(self, parser, namespace, values, option_string=None):
      print('Action Namespace=%r values=%r option_string=%r' % (namespace, values, option_string))
      setattr(namespace, self.dest, values) # do the normal attr set for dest
      prescan = prescan_sweep_dirs(values)
      setattr(namespace, 'prescan',prescan)
      
def main(argv):
   # parser = argparse.ArgumentParser(description='Produce Sweep Graphs for RAJAPerf output saved in one or more directories')
   # kgroup = parser.add_mutually_exclusive_group()
   # kgroup.add_argument('--kernels', nargs='+',
   #                     help='kernels to include')
   # kgroup.add_argument('--exclude_kernels', nargs="+",
   #                     help='kernels to exclude')
   # vgroup = parser.add_mutually_exclusive_group()
   # vgroup.add_argument('--variants', nargs='+',
   #                     help='variants to include')
   # vgroup.add_argument('--exclude_variants', nargs="+",
   #                     help='variants to exclude')
   # parser.add_argument('--split_line_graph', choices=['time(s)','time(ms)','time(us)'])
   # parser.add_argument('--directories',nargs='+', action=DirectoryAction,
   #                     help='Directories to process')
   # parser.print_help()
   # #args = parser.parse_args('--kernels k1 k2 k3  --exclude_variants v1 v2 --split_line_graph time(s)'.split())
   # args = parser.parse_args(argv)
   # #args.candidate_kernels = ['k3', 'k4', 'k5']
   # print(args)

   parent_parser = argparse.ArgumentParser(add_help=False)
   parent_parser.add_argument('--directories',nargs='+',action=DirectoryAction)
   child_parser = argparse.ArgumentParser(parents=[parent_parser])
   kgroup = child_parser.add_mutually_exclusive_group()
   kgroup.add_argument('--kernels', nargs='+',action=KernelAction,
                        help='kernels to include')

   args = child_parser.parse_args(argv)
   print(args)
   
   
if __name__ == '__main__':
   main(sys.argv[1:])