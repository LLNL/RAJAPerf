raja_add_library(
  NAME apps
  SOURCES PRESSURE_CALC.cpp 
          ENERGY_CALC.cpp 
          VOL3D_CALC.cpp 
          DEL_DOT_VEC_2D.cpp 
          COUPLE.cpp
  DEPENDS_ON common RAJA)
