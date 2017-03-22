add_library(apps
  ENERGY_CALC.cxx
  PRESSURE_CALC.cxx
  VOL3D_CALC.cxx)

target_link_libraries(apps
  common
  RAJA)
