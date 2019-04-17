#pragma once
#define USE_CALIPER //TODO DZP: delete before PR gets merged
#ifdef USE_CALIPER
#include <caliper/Annotation.h>
#include "rajaperf_config.hpp"
#endif
template<typename T>
struct Recorder;
template<typename T>
struct Recorder{
  static cali::Annotation& record(std::string key, T value, int opts){
#ifdef USE_CALIPER
    return cali::Annotation(key.c_str() , opts).begin(value);
#endif
  }
};
template<>
struct Recorder<std::string>{
  static cali::Annotation& record(std::string key, std::string value, int opts){
#ifdef USE_CALIPER
    return cali::Annotation(key.c_str() , opts).begin(value.c_str());
#endif
  }
};
template<>
struct Recorder<long double>{
  static cali::Annotation& record(std::string key, long double value, int opts){
#ifdef USE_CALIPER
    return cali::Annotation(key.c_str() , opts).begin((double)value);
#endif
  }
};

template<typename T>
cali::Annotation& record(std::string key, T value, int opts){
  return Recorder<T>::record(key, value, opts);
}

template<typename T>
void declareMetadata(std::string key, T value) {
  record(key, value, CALI_ATTR_SKIP_EVENTS);
}

template<typename T>
void declarePerformanceResult(std::string kernel_name, std::string variant_name, std::string measurement_name, T value){
#ifdef USE_CALIPER
  cali::Annotation result_annot = record(kernel_name, value, 0);
  declareMetadata(key, value);
  result_annot.end();
#endif
}

