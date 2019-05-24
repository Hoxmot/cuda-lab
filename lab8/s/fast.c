#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 100
#define N 100

#define max(a, b) ((a) > (b) ? (a) : (b))

double error(double err, double A, double B)
{
	double pom;
	pom = abs(A - B);
	/*
    if (pom < 0)
		pom = -pom;
    
	if (err > pom)
		return err;
	return pom;
    */
    return max(pom, err);
}

void startA (int m, int n, double** A){
	int i, j;
	for (i = 0; i < m; i++) {
		A[0][i] = 100;
		A[1][i] = 100;
  		for (j = 2; j < n; j++){
			A[j][i] = 10;
		}
	}
}

void makeMV(int n, int m, double** A, double** ANEW, double* err){
	int i, j;
	*err = 0;
	for (i=1; i<m-1; i++)
		for (j=1; j<n-1; j++){
			ANEW[j][i] = (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]) * 0.25;
            double pom = abs(A[j][i] - ANEW[j][i]);
			//*err = *err > pom ? *err : pom;
            *err = error(*err, A[j][i], ANEW[j][i]);
		}
}

void copy(int n, int m, double** A, double** ANEW){
	int i, j;
	for (j = 1; j < n-1; j++)
        for (i = 1; i < m-1; i++)
			A[j][i] = ANEW[j][i];
}

int main() {
    return 0;
    int i, j;
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
