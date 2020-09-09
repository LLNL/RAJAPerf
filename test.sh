#!/bin/bash

# Clang 9
run_clang9_diagnostic () {
    ./scripts/lc-builds/toss3_clang9.0.0.sh orig
    cd build_lc_toss3-clang-9.0.0_orig
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_clang9.0.0.sh origExt
    cd build_lc_toss3-clang-9.0.0_origExt
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_clang9.0.0.sh dev
    cd build_lc_toss3-clang-9.0.0_dev
    make -j 36
    cd ../
    
    for i in 1 2 3
    do
      #clang: run the original v0.11.0
      echo "CLANG: Running the original"
      cd build_lc_toss3-clang-9.0.0_orig
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    
      #clang: run the originalExt v0.11.0
      echo "CLANG: Running the originalExt"
      cd build_lc_toss3-clang-9.0.0_origExt
      srun -n 1 ./bin/raja-perf.exe
      cd ../
      
      #clang: run the vec v0.11.0
      echo "CLANG: Running the dev"
      cd build_lc_toss3-clang-9.0.0_dev
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    done
}

#GCC 8.1
run_gcc81_diagnostic () {
    ./scripts/lc-builds/toss3_gcc8.1.0.sh orig
    cd build_lc_toss3-gcc-8.1.0_orig
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_gcc8.1.0.sh origExt
    cd build_lc_toss3-gcc-8.1.0_origExt
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_gcc8.1.0.sh dev
    cd build_lc_toss3-gcc-8.1.0_dev
    make -j 36
    cd ../

    for i in 1 2 3
    do
      #gcc: run the original v0.11.0
      echo "GCC: Running the original"
      cd build_lc_toss3-gcc-8.1.0_orig 
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    
      #gcc: run the originalExt v0.11.0
      echo "GCC: Running the originalExt"
      cd build_lc_toss3-gcc-8.1.0_origExt
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    
      #gcc: run the dev v0.11.0
      echo "GCC: Running the dev"
      cd build_lc_toss3-gcc-8.1.0_dev
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    done
}

#ICPC 19.1
run_icpc191_diagnostic () {
    ./scripts/lc-builds/toss3_icpc19.1.0.sh orig
    cd build_lc_toss3-icpc-19.1.0_orig
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_icpc19.1.0.sh origExt
    cd build_lc_toss3-icpc-19.1.0_origExt
    make -j 36
    cd ../
    
    ./scripts/lc-builds/toss3_icpc19.1.0.sh dev
    cd build_lc_toss3-icpc-19.1.0_dev
    make -j 36
    cd ../
    
    for i in 1 2 3
    do
      #icpc: run the original v0.11.0
      echo "ICPC: Running the original"
      cd build_lc_toss3-icpc-19.1.0_orig 
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    
      #gcc: run the originalExt v0.11.0
      echo "ICPC: Running the originalExt"
      cd build_lc_toss3-icpc-19.1.0_origExt
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    
      #ICPC: run the dev v0.11.0
      echo "ICPC: Running the dev"
      cd build_lc_toss3-icpc-19.1.0_dev
      srun -n 1 ./bin/raja-perf.exe
      cd ../
    done

}

#run_clang9_diagnostic
#run_gcc81_diagnostic
run_icpc191_diagnostic
