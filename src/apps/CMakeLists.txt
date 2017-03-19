add_library(apps
  PRESSURE_CALC.cxx)

target_link_libraries(apps
  common
  RAJA)
