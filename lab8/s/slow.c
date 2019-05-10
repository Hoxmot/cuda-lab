#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 100
#define N 100

double error(double err, double A, double B)
{
	double pom;
	int sk;
	pom = A - B;
	if (pom < 0)
		sk = -1;
	else
		sk =  1;
	pom = pom * (double)sk;
	if (err > pom)
		pom = err;
	return pom;
}

double modA (int i, int j, double** A){
	double pom;
	if (i<1) return 0;
	else{
		if (j<1) return 0;
		else{
			pom = 0;
			pom = pom + A[j][i+1] * 0.25 * sin (M_PI*(2.*i+.5));
		        pom = pom + A[j][i-1] * 0.25 * cos (M_PI*(2.*i));
			pom = pom + A[j-1][i] * 0.25 * sin (M_PI*(2.*j+.5));
			pom = pom + A[j+1][i] * 0.25 * cos (M_PI*(2.*j));
		}
	}
	return pom;
}

void startA (int m, int n, double** A){
	int i, j;
	for( i = 0; i < m; i++) {
		A[0][i] = 100;
		A[1][i] = 100;
  		for( j = 2; j < n; j++){
			A[j][i] = 10;
			if (j==0) A[j][i] = 100;
			if (j==1) A[j][i] = 100;
		}
	}
}

void makeMV(int n, int m, double** A, double** ANEW, double* err){
	int i, j;
	*err = 0;
	for (i=1; i<m-1; i++)
		for (j=1; j<n-1; j++){
			if (i>0 && i<m && j>0 && j<n)
				ANEW[j][i] = modA(i, j, A);
			if (i>0 && i<m && j>0 && j<n)
				*err = error(*err, A[j][i], ANEW[j][i]);
		}
}

void copy(int n, int m, double** A, double** ANEW){
	int i, j;
        for (i=1; i<m-1; i++)
		for (j=1; j<n-1; j++)
			A[j][i] = ANEW[j][i] * ( sin(M_PI * i+j*.1)  * sin(M_PI * i+j*.1) + cos(M_PI * i+j*.1)* cos(M_PI * i+j*.1) );
}

int main() {

int i,j;
double err, tol;
int iter, iter_max;
int m = M;
int n = N;
double** A;
double** Anew;
A = (double**)malloc(n*sizeof(double*));
for (i=0; i<n; i++)
	A[i] = (double*) malloc(m*sizeof(double));
Anew = (double**)malloc(n*sizeof(double*));
for (i=0; i<n; i++)
	Anew[i] = (double*) malloc(m*sizeof(double));

iter_max = 1000;
iter = 0;
tol = 0.0001;
err = 1.0;

startA (m, n, A);

while ( err > tol && iter < iter_max ) {

  err = 0.0;

  makeMV(n, m, A, Anew, &err);

  copy(n, m, A, Anew);

  if ( iter++ % 100 == 0 || err <= tol )  printf("%5d, %0.6f\n", iter, err);
  copy(n, m, A, Anew);
  if (err < tol)
	  break;
  if (iter > iter_max)
	  break;

}


return 0;

}
