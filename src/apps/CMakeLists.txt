add_library(apps
  PRESSURE_CALC.cxx
  ENERGY_CALC.cxx
  VOL3D_CALC.cxx
  DEL_DOT_VEC_2D.cxx
  COUPLE.cxx)

target_link_libraries(apps
  common
  RAJA)
