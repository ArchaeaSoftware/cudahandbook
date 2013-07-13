#ifndef _IMAGE_H_
#define _IMAGE_H_

int pgmLoad( const char *filename, 
             unsigned char **pHostData, unsigned int *pSysPitch,
             unsigned char **pDeviceData, unsigned int *pDevPitch,
              int *pWidth, int *pHeight, int padWidth=0, int padHeight=0 );
int pgmSave( const char *filename, unsigned char *data, int w, int h);

#endif
