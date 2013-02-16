CXX := g++

CFLAGS := -O3
CXXFLAGS := -fopenmp -pthread
LDFLAGS := -fopenmp -pthread -lrt

ifneq ($(NO_CUDA),)
NVCC := g++
NVCCFLAGS := -xc++
CFLAGS += -DNO_CUDA
else
NVCC := nvcc
NVCCFLAGS :=
LDFLAGS += -lcudart
endif

LINK := $(CXX)

INCLUDE := -I../chLib

SOURCES := \
	nbody.cu \
	nbody_CPU_SSE.cpp \
	nbody_CPU_SSE_threaded.cpp \
	nbody_CPU_SSE_openmp.cpp

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

all: nbody

clean:
	rm -f nbody
	rm -f $(OBJECTS)

%.o: %.cu .cflags
	$(QUIET_NVCC)$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cpp .cflags
	$(QUIET_CXX)$(CXX) $(CFLAGS) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

nbody: $(OBJECTS)
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CFLAGS = $(subst ','\'',$(CC) $(CFLAGS) $(LDFLAGS))

.cflags: .force-cflags
	@FLAGS='$(TRACK_CFLAGS)'; \
	if test x"$$FLAGS" != x"`cat .cflags 2>/dev/null`" ; then \
		echo "    * rebuilding nbody: new build flags or prefix"; \
		echo "$$FLAGS" > .cflags; \
	fi

.PHONY: .force-cflags

endif
