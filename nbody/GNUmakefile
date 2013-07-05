BIN := nbody

all: $(BIN)

uname_S := $(shell uname -s 2>/dev/null || echo "not")
uname_M := $(shell uname -m 2>/dev/null || echo "not")
uname_O := $(shell uname -o 2>/dev/null || echo "not")

CXX := g++
NVCC := nvcc

DEBUG = 0

NVCCFLAGS := -O3
CFOPTIMIZE := -O3

ifneq ($(DEBUG),0)
CFOPTIMIZE := -O0 -ggdb
NVCCFLAGS := -O0 -g
endif

CFLAGS :=
CXXFLAGS := $(CFOPTIMIZE) -fno-strict-aliasing

ifeq ($(NO_OPENMP),)
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

ifeq ($(NO_SIMD),)
ifeq ($(uname_M),ppc)
CFLAGS += -DHAVE_ALTIVEC
CXXFLAGS += -maltivec
endif
ifeq ($(uname_M),armv7l)
CFLAGS += -DHAVE_NEON
CXXFLAGS += -marm -mfpu=neon -ffast-math
endif
ifeq ($(uname_M),x86_64)
CFLAGS += -DHAVE_SSE
endif
endif

ifneq ($(DEBUG),0)
CFLAGS += -DDEBUG=1
endif

CUDA_ROOT := $(dir $(shell which nvcc 2>/dev/null))..

ifneq ($(uname_O),Cygwin)
CXXFLAGS += -pthread
LDFLAGS += -pthread
endif

ifeq ($(uname_S),Linux)
LDFLAGS += -lrt
endif

ifeq ($(CXX),icpc)
# Special Intel compiler options to give us more detail
CXXFLAGS += -parallel -openmp-report2 -vec-report2 -par-report2

# Statically link Intel libraries so the executables are more portable
LDFLAGS += -static-intel
endif

ifneq ($(NO_CUDA),)
NVCC := g++
NVCCFLAGS += -xc++
CFLAGS += -DNO_CUDA
else
NVCCFLAGS +=
LDFLAGS += -L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib64 -lcudart
endif

LINK := $(CXX)

INCLUDE := -I../chLib

SOURCES := \
	nbody.cu \
	nbody_CPU_AOS.cpp \
	nbody_CPU_AOS_tiled.cpp \
	nbody_CPU_SOA.cpp \
	nbody_CPU_AltiVec.cpp \
	nbody_CPU_NEON.cpp \
	nbody_CPU_SSE.cpp \
	nbody_CPU_SSE_threaded.cpp

ifeq ($(NO_OPENMP),)
SOURCES += \
	nbody_CPU_AltiVec_openmp.cpp \
	nbody_CPU_NEON_openmp.cpp \
	nbody_CPU_SSE_openmp.cpp
endif

ifeq ($(NO_CUDA),)
SOURCES += \
	nbody_GPU_shared.cu \
	nbody_multiGPU.cu \
	nbody_multiGPU_threaded.cu
endif

OBJECTS := $(SOURCES:%.cu=%.o)
OBJECTS := $(OBJECTS:%.cpp=%.o)

ifneq ($(findstring $(MAKEFLAGS),s),s)
ifndef V
        QUIET_CC        = @echo '   ' CC   $@;
        QUIET_CXX       = @echo '   ' CXX  $@;
        QUIET_NVCC      = @echo '   ' NVCC $@;
        QUIET_LINK      = @echo '   ' LD   $@;
        QUIET           = @
        export V
endif
endif

clean:
	rm -f $(BIN)
	rm -f $(OBJECTS)

%.o: %.cu .cflags-$(BIN)
	$(QUIET_NVCC)$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cpp .cflags-$(BIN)
	$(QUIET_CXX)$(CXX) $(CFLAGS) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

$(BIN): $(OBJECTS)
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CFLAGS = $(subst ','\'',$(CXX) $(NVCC) $(LINK) $(CFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDE) $(LDFLAGS))

.cflags-$(BIN): .force-cflags
	@FLAGS='$(TRACK_CFLAGS)'; \
	if test x"$$FLAGS" != x"`cat .cflags-$(BIN) 2>/dev/null`" ; then \
		echo "    * rebuilding $(BIN): new build flags or prefix"; \
		echo "$$FLAGS" > .cflags-$(BIN); \
	fi

.PHONY: .force-cflags

endif
