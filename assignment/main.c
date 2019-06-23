#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

float dyst(float* positionX, float* positionY, float* positionZ , int i, int j){
	float X = positionX[i] - positionX[j];
	float Y = positionY[i] - positionY[j];
	float Z = positionZ[i] - positionZ[j];
	return (X*X + Y*Y + Z*Z);
}

float Energy(float* positionX, float* positionY, float* positionZ, int i, int N){
	int j;
	float E = 0;
	float d;
	for (j=0; j<N; j++){
		if (i!=j){
			d = dyst(positionX, positionY, positionZ , i, j);
			E += 1./d;
		}
	}
	return E;
}

float RAND1(){
	return (rand()/(1.+RAND_MAX) - 0.5); 
}
float RAND0(){
	return rand()/(1.+RAND_MAX);
}

void makemove(float* positionX, float* positionY, float* positionZ, float* positionNEWX, float* positionNEWY, float* positionNEWZ, int N){
	memcpy(positionX, positionNEWX, N*sizeof(float));
	memcpy(positionY, positionNEWY, N*sizeof(float));
	memcpy(positionZ, positionNEWZ, N*sizeof(float));
}
void newpos(float* positionNEWX, float* positionNEWY, float* positionNEWZ, int i, int N){
	float size = cbrt(N);
	positionNEWX[i] += RAND1();
	if (positionNEWX[i]<0) positionNEWX[i] = fabs(positionNEWX[i]);
	else if (positionNEWX[i] > size) positionNEWX[i] -= size;
	positionNEWY[i] += RAND1();
        if (positionNEWY[i]<0) positionNEWY[i] = fabs(positionNEWY[i]);
        else if (positionNEWY[i] > size) positionNEWY[i] -= size;
	positionNEWZ[i] += RAND1();
        if (positionNEWZ[i]<0) positionNEWZ[i] = fabs(positionNEWZ[i]);
        else if (positionNEWZ[i] > size) positionNEWZ[i] -= size;
}

float step (float* positionX, float* positionY, float* positionZ, float* positionNEWX, float* positionNEWY, float* positionNEWZ, int N, float kT){
	int i, k;
	float* pom;
	float E;
	int stepY = 0;
	for (i=0; i<N; i++){
		newpos(positionNEWX, positionNEWY, positionNEWZ, i, N);
		E = Energy(positionNEWX, positionNEWY, positionNEWZ, i, N) - Energy(positionX, positionY, positionZ, i, N);
		if (E<0){
			makemove(positionX, positionY, positionZ, positionNEWX, positionNEWY, positionNEWZ, N);
			stepY++;
		}
		else if(RAND0() < exp(-E/kT)){
			makemove(positionX, positionY, positionZ, positionNEWX, positionNEWY, positionNEWZ, N);
			stepY++;
		}
	}
	return stepY * 1./N;
}

void start (float* positionX, float* positionY, float* positionZ, float* positionNEWX, float* positionNEWY, float* positionNEWZ, int N){
	float size = cbrt(N);
	int i;
	for (i=0; i<N; i++){
		positionX[i] = RAND0() * size;
		positionY[i] = RAND0() * size;
		positionZ[i] = RAND0() * size;
		positionNEWX[i] = positionX[i];
		positionNEWY[i] = positionY[i];
		positionNEWZ[i] = positionZ[i];
	}
}

void pr(float* positionX, float* positionY, float* positionZ, int N){
	int i;
	for (i=0; i<N; i++)
		printf("%f %f %f\n", positionX[i], positionY[i], positionZ[i]);
	printf("\n\n");
}

int main(){
   int const N = 100;
   int k, T;
   int steps=1000;
   float kT;
   float stepY;
   float* positionX = (float*) malloc(N * sizeof(float));
   float* positionY = (float*) malloc(N * sizeof(float));
   float* positionZ = (float*) malloc(N * sizeof(float));
   float* positionNEWX = (float*) malloc(N * sizeof(float));
   float* positionNEWY = (float*) malloc(N * sizeof(float));
   float* positionNEWZ = (float*) malloc(N * sizeof(float));
   for (T=270; T<360; T+=10){
	stepY = 0;
	kT = .01/T;
   	start (positionX, positionY, positionZ, positionNEWX, positionNEWY, positionNEWZ, N);
   	for (k=0; k<steps; k++){
	   stepY += step(positionX, positionY, positionZ, positionNEWX, positionNEWY, positionNEWZ, N, kT);
	   pr(positionX, positionY, positionZ, N);
   	}
	printf("Stepe ACC %d  %f\n", T, stepY*1./steps);
   }
   return 0;
}

