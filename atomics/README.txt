This directory contains sample code with no purpose other than to
illustrate the code generated for atomic operations on global and
shared memory.  

To see the generated machine code, compile with:

nvcc --cubin --gpu-architecture sm_xx <filename>

then disassemble the compiled microcode with cuobjdump:

cuobjdump --dump-sass <filename>.cubin


Memory     Operand              File

Global     int                  atomic32.cu
Global     unsigned long long   atomic64.cu
Global     float                atomicFloat.cu

Shared     int                  atomic32Shared.cu
Shared     unsigned long long   atomic64Shared.cu
Shared     float                atomicFloatShared.cu

