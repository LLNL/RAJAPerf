#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors. 
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#=============================================================================
# Change the copyright date in all files that contain the text
# "the RAJAPerf/COPYRIGHT file", which is part of the copyright statement
# at the top of each RAJA file. We use this to distinguish RAJA files from
# that we do not own (e.g., other repos included as submodules), which we do
# not want to modify. Note that this file and *.git files are omitted
# as well.
#
# IMPORTANT: Since this file is not modified (it is running the shell 
# script commands), you must EDIT THE COPYRIGHT DATES ABOVE MANUALLY.
#
# Edit the 'find' command below to change the set of files that will be
# modified.
#
# Change the 'sed' command below to change the content that is changed
# in each file and what it is changed to.
#
#=============================================================================
#
# If you need to modify this script, you may want to run each of these 
# commands individual from the command line to make sure things are doing 
# what you think they should be doing. This is why they are seperated into 
# steps here.
# 
#=============================================================================

#=============================================================================
# First find all the files we want to modify
#=============================================================================
find . -type f ! -name \*.git\*  ! -name \*update_copyright\* -exec grep -l "the RAJAPerf/COPYRIGHT file" {} \; > files2change

#=============================================================================
# Replace the old copyright dates with new dates
#=============================================================================
for i in `cat files2change`
do
    echo $i
    cp $i $i.sed.bak
    sed "s/Copyright (c) 2017-20/Copyright (c) 2017-21/" $i.sed.bak > $i
done

for i in LICENSE RELEASE README.md
do
    echo $i
    cp $i $i.sed.bak
    sed "s/2017-2020/2017-2021/" $i.sed.bak > $i
done

#=============================================================================
# Remove temporary files created in the process
#=============================================================================
find . -name \*.sed.bak -exec rm {} \;
rm files2change
