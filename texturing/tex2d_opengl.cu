/*
 *
 * tex2d_opengl.cu
 *
 * Microdemo to illustrate the workings of 2D texturing.
 *
 * Build with: nvcc -I ../chLib <options> tex2d_opengl.cu, with
 * platform-specific OpenGL includes and libs.
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
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
#include <ctype.h>

#include <chError.h>

#include <GL/glut.h>

int g_width, g_height;
float g_scale = 1.0f;
char *g_texture;

cudaArray *g_arrayTexture;
uchar4 *g_hostFrameBuffer;
uchar4 *g_deviceFrameBuffer;
bool g_displayText = true;

char *
LoadTexturePPM( const char * filename )
{
    int width, height, maxval;
    FILE * file = NULL;
    char *data = NULL;

    // open texture data
    file = fopen( filename, "rb" );
    if ( file == NULL )
        goto Error;

    if ( 'P' != fgetc( file ) )
        goto Error;
    if ( '6' != fgetc( file ) )
        goto Error;
    if ( 1 != fscanf( file, "%d", &width ) )
        goto Error;
    if ( 1 != fscanf( file, "%d", &height ) )
        goto Error;
    if ( 1 != fscanf( file, "%d", &maxval ) )
        goto Error;
    if ( maxval != 0xff )
        goto Error;

    {
        int ch;
        do {
            ch = fgetc( file );
        } while ( isspace( ch ) );
    }

    // allocate buffer
    data = (char *) malloc( width * height * 3 );
    if ( ! data )
        goto Error;

    // read texture data
    if ( 1 != fread( data, width * height * 3, 1, file ) )
        goto Error;

    fclose( file );
    return data;
Error:
    if ( file ) {
        fclose( file );
    }
    free( data );
    return NULL;
}

texture<uchar4, 2, cudaReadModeElementType> tex2d;

cudaError_t
CreateAndPopulateArray( cudaArray **ret, char *base, int width, int height )
{
    uchar4 *array4 = new uchar4[width*height];
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaError_t status = cudaMallocArray( ret, &desc, width, height );
    if ( cudaSuccess != status ) {
        return status;
    }
    for ( int row = 0; row < 256; row++ ) {
        char *baserow = base+(height-row-1)*3*width;
        for ( int col = 0; col < 256; col++ ) {
            uchar4 value;
            value.z = *baserow++;
            value.y = *baserow++;
            value.x = *baserow++;
            value.w = 0;
            array4[row*width+col] = value;
        }
    }

    return cudaMemcpy2DToArray( *ret, 0, 0, array4, width*4, width*4, height, cudaMemcpyHostToDevice );
}

__global__ void
RenderTextureUnnormalized( uchar4 *out, int width, int height )
{
    for ( int j = blockIdx.x; j < height; j += gridDim.x ) {
        int row = height-j-1;
        for ( int col = threadIdx.x; col < width; col += blockDim.x ) {
            out[row*width+col] = tex2D( tex2d, (float) col, (float) row );
        }
    }
}

__global__ void
RenderTextureNormalized( uchar4 *out, 
                         int width, 
                         int height, 
                         int scale )
{
    for ( int j = blockIdx.x; j < height; j += gridDim.x ) {
        int row = height-j-1;
        out = (uchar4 *) (((char *) out)+row*4*width);
        float texRow = scale * (float) row / (float) height;
        float invWidth = scale / (float) width;
        for ( int col = threadIdx.x; col < width; col += blockDim.x ) {
            float texCol = col * invWidth;
            out[col] = tex2D( tex2d, texCol, texRow );
        }
    }
}

void glPrint(int x, int y, const char *s, void *font)
{
    glColor3f(1.0, 1.0, 1.0);
    
    glRasterPos2f(x, y);
    int len = (int) strlen(s);
    for (int i = 0; i < len; i++) {
        glutBitmapCharacter(font, s[i]);
    }
}

int g_idxAddressMode;

void displayCB(void)		/* function called whenever redisplay needed */
{
    glClear(GL_COLOR_BUFFER_BIT);

    if ( cudaSuccess != cudaBindTextureToArray( tex2d, 
                                                g_arrayTexture, 
                                                cudaCreateChannelDesc<uchar4>() ) )
    {
        return;
    }
    if ( tex2d.normalized ) {
        RenderTextureNormalized<<<g_height, 384>>>( g_deviceFrameBuffer, g_width, g_height, g_scale );
    }
    else {
        RenderTextureUnnormalized<<<g_height, 384>>>( g_deviceFrameBuffer, g_width, g_height );
    }
    if ( cudaSuccess != cudaDeviceSynchronize() )
        return;
    glRasterPos2f( 0.0f, 0.0f );
    glDrawPixels( g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, g_hostFrameBuffer );
    if ( g_displayText ) {
        char s[256];
        int fontWidth = 9;      // width of 9x15 font
        int x = fontWidth;      // one character of 9x15 font
        int fontHeight = 15;

        int y = g_height - fontHeight;
        if ( tex2d.normalized ) {
            glPrint( x, y, "Normalized (hit N for unnormalized)", GLUT_BITMAP_9_BY_15 );
        }
        else {
            glPrint( x, y, "Unnormalized (hit N for normalized)", GLUT_BITMAP_9_BY_15 );
        }
        y -= fontHeight;
        switch ( tex2d.addressMode[0] ) {
            case cudaAddressModeClamp:
                sprintf( s, "X address mode: Clamp %s", g_idxAddressMode ? "\0" :
                    "(W=wrap, M=mirror, B=border)" );
                break;
            case cudaAddressModeWrap:
                sprintf( s, "X address mode: Wrap %s", g_idxAddressMode ? "\0" :
                    "(C=clamp, M=mirror, B=border)" );
                break;
            case cudaAddressModeMirror:
                sprintf( s, "X address mode: Mirror %s", g_idxAddressMode ? "\0" :
                    "(C=clamp, W=wrap, B=border)" );
                break;
            case cudaAddressModeBorder:
                sprintf( s, "X address mode: Border %s", g_idxAddressMode ? "\0" :
                    "(C=clamp, W=wrap, M=mirror)" );
                break;
        }
        glPrint( x, y, s, GLUT_BITMAP_9_BY_15 );
        y -= fontHeight;

        switch ( tex2d.addressMode[1] ) {
            case cudaAddressModeClamp:
                sprintf( s, "Y address mode: Clamp %s", !g_idxAddressMode ? "\0" :
                    "(W=wrap, M=mirror, B=border)" );
                break;
            case cudaAddressModeWrap:
                sprintf( s, "Y address mode: Wrap %s", !g_idxAddressMode ? "\0" :
                    "(C=clamp, M=mirror, B=border)" );
                break;
            case cudaAddressModeMirror:
                sprintf( s, "Y address mode: Mirror %s", !g_idxAddressMode ? "\0" :
                    "(C=clamp, W=wrap, B=border)" );
                break;
            case cudaAddressModeBorder:
                sprintf( s, "Y address mode: Border %s", !g_idxAddressMode ? "\0" :
                    "(C=clamp, W=wrap, M=mirror)" );
                break;
        }
        glPrint( x, y, s, GLUT_BITMAP_9_BY_15 );
        y -= fontHeight;
        glPrint( x, y, "Hit X to set X addressing mode, Y to set Y addressing mode", GLUT_BITMAP_9_BY_15 );
        y -= fontHeight;
        glPrint( x, y, "When in normalized mode, hit 1-9 keys to set scale", GLUT_BITMAP_9_BY_15 );
        y -= fontHeight;
        glPrint( x, y, "T key toggles text display", GLUT_BITMAP_9_BY_15 );
    }
    glFlush();				/* Complete any pending operations */
}

void keyCB(unsigned char key, int x, int y)	/* called on key press */
{
    switch ( key ) {
        case '1': case '2': case '3':
        case '4': case '5': case '6':
        case '7': case '8': case '9':
            g_scale = (float) (key-'0');
            break;
        case 'q':
            exit(0);
            break;
        case 'x':
            g_idxAddressMode = 0;
            break;
        case 'y':
            g_idxAddressMode = 1;
            break;
        case 'n':
            tex2d.normalized = ! tex2d.normalized;
            break;
        case 'w':
            tex2d.addressMode[g_idxAddressMode] = cudaAddressModeWrap;
            break;
        case 'c':
            tex2d.addressMode[g_idxAddressMode] = cudaAddressModeClamp;
            break;
        case 'm':
            tex2d.addressMode[g_idxAddressMode] = cudaAddressModeMirror;
            break;
        case 'b':
            tex2d.addressMode[g_idxAddressMode] = cudaAddressModeBorder;
            break;
        case 't':
            g_displayText = ! g_displayText;
            break;
        default: 
            return;
    }
    glutPostRedisplay();
}

void
reshapeCB( int width, int height )
{
    cudaError_t status;

    if ( g_hostFrameBuffer ) {
        cudaFreeHost( g_hostFrameBuffer );
    }

    g_width = width;
    g_height = height;

    status = cudaHostAlloc( &g_hostFrameBuffer, g_width*sizeof(uchar4)*g_height, cudaHostAllocMapped );
    if ( cudaSuccess != status ) {
        goto Error;
    }
    status = cudaHostGetDevicePointer( &g_deviceFrameBuffer, g_hostFrameBuffer, 0 );
    if ( cudaSuccess != status ) {
        goto Error;
    }
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluOrtho2D(0,width,0,height);
    glViewport(0,0,width,height);
    glutPostRedisplay();
Error:;
}

int
main(int argc, char *argv[])
{
    cudaError_t status;
    int ret = 1;

    g_width = 512;
    g_height = 512;

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(g_width,g_height);
    glutCreateWindow("CUDA 2D Texturing");

    g_texture = LoadTexturePPM( "TextureDemoImage.ppm" );
    if ( ! g_texture ) {
        fprintf( stderr, "Could not load texture\n");
        goto Error;
    }

    cuda(SetDeviceFlags( cudaDeviceMapHost ) );

    CUDART_CHECK( CreateAndPopulateArray( &g_arrayTexture, g_texture, 256, 256 ) );

    glClearColor(0.0,0.0,0.0,0.0);
    glutDisplayFunc(displayCB);
    glutKeyboardFunc(keyCB);
    glutReshapeFunc(reshapeCB);

    glutMainLoop();

    // we never get here

    ret = 0;
Error:
    return ret;
}
