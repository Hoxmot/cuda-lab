//
//  Mandelbrot.h
//  
//
//  Created by Witold Rudnicki on 22.05.2015.
//
//

#ifndef ____Mandelbrot__
#define ____Mandelbrot__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#endif /* defined(____Mandelbrot__) */

int computeMandelbrot(double X0, double Y0, double X1, double Y1, int POZ, int PION, int ITER, int *Mandel);
void makePicture(int *Mandel, int width, int height, int MAX);
void makePictureInt(int *Mandel,int width, int height, int MAX);
void dumpMandel(int *Mandel, int POZ, int PION);
