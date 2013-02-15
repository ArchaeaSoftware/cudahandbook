CC = nvcc
CFLAGS = -O3
INCLUDE = -I../chLib
LDFLAGS = -lpthread -lrt

SOURCES = nbody.cu nbody_CPU_SSE.cpp nbody_CPU_SSE_threaded.cpp nbody_GPU_shared.cu nbody_multiGPU.cu nbody_multiGPU_threaded.cu
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

%.o: %.cu
	$(QUIET_CC)$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cpp
	$(QUIET_CC)$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

nbody: $(OBJECTS)
	$(QUIET_LINK)$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(OBJECTS) $(LDFLAGS)
