//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_CaliperUtil_HPP
#define RAJAPerf_CaliperUtil_HPP

#ifdef USE_CALIPER
#include <caliper/Annotation.h>
#include "rajaperf_config.hpp"
#endif
#ifdef USE_CALIPER
template<typename T>
struct Recorder;
template<typename T>
struct Recorder{
  static cali::Annotation& record(std::string key, T value, int opts){
    return cali::Annotation(key.c_str() , opts).begin(value);
  }
};
template<>
struct Recorder<std::string>{
  static cali::Annotation& record(std::string key, std::string value, int opts){
    return cali::Annotation(key.c_str() , opts).begin(value.c_str());
  }
};
template<>
struct Recorder<long double>{
  static cali::Annotation& record(std::string key, long double value, int opts){
    return cali::Annotation(key.c_str() , opts).begin((double)value);
  }
};

template<typename T>
cali::Annotation& record(std::string key, T value, int opts){
  return Recorder<T>::record(key, value, opts);
}
#endif // USE_CALIPER

template<typename T>
void declareMetadata(std::string key, T value) {
#ifdef USE_CALIPER
  record(key, value, CALI_ATTR_SKIP_EVENTS);
#endif // USE_CALIPER
}

template<typename T>
void declarePerformanceResult(std::string kernel_name, std::string variant_name, std::string measurement_name, T value){
#ifdef USE_CALIPER
  cali::Annotation result_annot = record(kernel_name, value, 0);
  declareMetadata(kernel_name+"#"+variant_name+"_"+measurement_name, value);
  result_annot.end();
#endif // USE_CALIPER
}

#endif // end header guard
