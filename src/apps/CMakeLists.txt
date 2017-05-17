raja_add_library(
  NAME apps
  SOURCES PRESSURE.cpp 
          ENERGY.cpp 
          VOL3D.cpp 
          DEL_DOT_VEC_2D.cpp 
          COUPLE.cpp
          FIR.cpp
  DEPENDS_ON common RAJA)
