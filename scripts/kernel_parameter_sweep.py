import numpy as np
import os
import fileinput
import sys


if __name__ == '__main__':
    block_sizes = [64, 128, 256, 512, 1024]
    grid_sizes = [2,4,8,16,32,64,128,256,512,1024,2**11,2**12,2**13,2**14,2**15,2**16,2**17,2**18,2**19,2**20]
    problem_sizes = [2**11,2**12,2**13,2**14,2**15,2**16,2**17,2**18,2**19,2**20]

    sizeof = 4.0 #floats
    checksum_threshold = 1.0e-6
    test = 'Apps_PRESSURE'
    variant = 'Base_HIP'
    
    header = "blocksize, gridsize, problem size, time (s), throughput (bytes/s), checksum"
    data_array = np.array([], dtype=np.int64).reshape(0,6)
    
    top_path = os.path.abspath(os.path.join(__file__ ,"../.."))
    script_path = os.path.abspath(os.path.join(__file__ ,".."))
    build_path = top_path + "/build"
    src_path = top_path + "/src/apps"
    test_file = "PRESSURE-Hip.cpp"
    exe_path = top_path + "/build/bin"
    make_cmd = 'make'
    git_cmd = 'git checkout ' + src_path + '/' + test_file

    cmd = "./raja-perf.exe -k " + test  + " --size"
    
    timing_file = "RAJAPerf-timing.csv"
    checksum_file = "RAJAPerf-checksum.txt"
    
    os.system(git_cmd)

    for grid_size in grid_sizes:
        for line in fileinput.input([src_path+"/"+test_file], inplace=True):
           if "const size_t grid_size" in line:
               line = '       const size_t grid_size = '+ str(grid_size) + ';\n'
           sys.stdout.write(line)
        for block_size in block_sizes:
            for line in fileinput.input([src_path+"/"+test_file], inplace=True):
               if "const size_t block_size" in line:
                   line = '  const size_t block_size = '+ str(block_size) + ';\n'
               sys.stdout.write(line)                   
            #Recompile
            os.chdir(build_path)
            os.system(make_cmd)
            os.chdir(exe_path)
            for problem_size in problem_sizes:
                print(block_size,grid_size,problem_size)
                os.system(cmd + " " + str(problem_size))
                with open(checksum_file, 'r') as f:
                    for line in f:
                        if line.startswith(variant):
                            checksum = float(line.split()[2].rstrip())                
        
                with open(timing_file, 'r') as f:
                    for line in f:
                        if line.startswith(test):
                            time = line.split(',')[4].rstrip()
                            throughput = (float(problem_size)*sizeof)/float(time)
                            if(checksum>checksum_threshold):
                                continue
                            array = np.array([int(block_size),int(grid_size),int(problem_size),float(time),float(throughput),float(checksum)])
                            data_array=np.vstack([data_array,array])
    
    print(header)
    print(data_array)
    os.chdir(script_path)
    np.savetxt("data.csv",data_array, header=header, fmt='%4d, %4d, %4d, %10.5f, %10.5f, %10.5f')
    print("Kernel parameter sweep results in data.csv")
