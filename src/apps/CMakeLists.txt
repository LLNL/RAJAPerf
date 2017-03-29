raja_add_library(
  NAME apps
  SOURCES PRESSURE_CALC.cxx 
          ENERGY_CALC.cxx 
          VOL3D_CALC.cxx 
          DEL_DOT_VEC_2D.cxx 
          COUPLE.cxx
  DEPENDS_ON common RAJA)
