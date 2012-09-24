/*
 *
 * radixSort.cu
 *
 * Microdemo and microbenchmark of Radix Sort.  CPU only for now.
 *
 * Build with: nvcc -I ../chLib <options> radixSort.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 

 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

template<const int b>
void
RadixPass( int *out, const int *in, size_t N, int shift, int mask )
{
    const int numCounts = 1<<b;
    int counts[numCounts];
    memset( counts, 0, sizeof(counts) );
    for ( size_t i = 0; i < N; i++ ) {
        int value = in[i];
        int index = (value & mask) >> shift;
        counts[index] += 1;
    }

    //
    // compute exclusive scan of counts
    //
    int sum = 0;
    for ( int i = 0; i < numCounts; i++ ) { 
        int temp = counts[i];
        counts[i] = sum;
        sum += temp;
    }

    //
    // scatter each input to the correct output
    //
    for ( size_t i = 0; i < N; i++ ) {
        int value = in[i];
        int index = (value & mask) >> shift;
        out[ counts[index] ] = value;
        counts[index] += 1;
    }
}

template<const int b>
int *
RadixSort( int *out[2], const int *in, size_t N )
{
    int shift = 0;
    int mask = (1<<b)-1;

    //
    // index of output array, ping-pongs between 0 and 1.
    //
    int outIndex = 0;

    RadixPass<b>( out[outIndex], in, N, shift, mask );
    while ( mask ) {
        outIndex = 1 - outIndex;
        shift += 1;
        mask <<= 1;
        RadixPass<b>( out[outIndex], out[1-outIndex], N, shift, mask );
    }
    return out[outIndex];

}

bool
TestSort( size_t N, int mask = 0 )
{
    bool ret = false;
    int *sortInput = new int[ N ];
    int *sortOutput[2];
    int *radixSortedArray = 0;
    std::vector<int> sortedOutput( N );
    sortOutput[0] = new int[ N ];
    sortOutput[1] = new int[ N ];

    if ( 0 == sortInput || 
         0 == sortOutput[0] ||
         0 == sortOutput[1] ) {
        goto Error;
    }

    if ( mask == 0 ) {
        mask = -1;
    }
    for ( int i = 0; i < N; i++ ) {
        sortedOutput[i] = sortInput[i] = rand() & mask;
    }

    {
        std::sort( sortedOutput.begin(), sortedOutput.end() );
    }

    //
    // RadixSort returns sortOutput[0] or sortOutput[1],
    // depending on where it wound up in the ping-pong
    // between output arrays.
    //
    radixSortedArray = RadixSort<1>( sortOutput, sortInput, N );


    for ( size_t i = 0; i < N; i++ ) {
        if ( radixSortedArray[i] != sortedOutput[i] ) {
#ifdef _WIN32
            _asm int 3
#endif
            goto Error;
        }
    }
    ret = true;
Error:
    delete[] sortInput;
    delete[] sortOutput[0];
    delete[] sortOutput[1];
    return ret;
}

int
main()
{

#define TEST_VECTOR( N, mask )  \
    if ( ! TestSort( N, mask ) ) {  \
        printf( "%s(N=%d, mask=%d) FAILED\n", "RadixSort<1>", (int) N, mask );  \
        exit(1);    \
    }

    TEST_VECTOR( 32, 0xf );

    TEST_VECTOR( 1048576, 0 );

}
