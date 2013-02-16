
ifneq ($(NO_CUDA),)
CC := g++
CFLAGS := -O3 -fopenmp -DNO_CUDA -pthread -xc++
LDFLAGS := -fopenmp -pthread -lrt
else
CC := nvcc
CFLAGS := -O3
LDFLAGS := -lpthread -lrt
endif

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
        QUIET_CC        = @echo '   ' CC $@;
        QUIET_LINK      = @echo '   ' LD $@;
        QUIET           = @
        export V
endif
endif

all: nbody

clean:
	rm -f nbody
	rm -f $(OBJECTS)

%.o: %.cu .cflags
	$(QUIET_CC)$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cpp .cflags
	$(QUIET_CC)$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

nbody: $(OBJECTS)
	$(QUIET_LINK)$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

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
