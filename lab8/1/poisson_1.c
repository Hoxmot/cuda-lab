#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void ewolucja_V (double** V_NEW, double** V, double** ro, int Nx, int Ny, double h, int kMAX, double epsilon)
{ 
 int k;
 double roznica;
 int x, y;
//poczatkowe
 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++)
         V[y][x] = sin(h*x)*sin(h*y);
//gestosc
 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++){
	 if (x==y && x==15)  ro[y][x] = 1;
	 else ro[y][x] = 0;
     }
//brzeg
 for (k=0; k<Ny; k++)
     V[k][0] = 0;
 for (k=0; k<Ny; k++)
     V[k][Nx-1] = 0;
 for (k=0; k<Nx; k++)
     V[0][k] = 0;
 for (k=0; k<Nx; k++)
     V[Ny-1][k] = 0;
//brzeg
 for (k=0; k<Ny; k++)
     V_NEW[k][0] = 0;
 for (k=0; k<Ny; k++)
     V_NEW[k][Nx-1] = 0;
 for (k=0; k<Nx; k++)
     V_NEW[0][k] = 0;
 for (k=0; k<Nx; k++)
     V_NEW[Ny-1][k] = 0;

 for (k=0; k<kMAX; k++) {
//krok
      for (y=1; y<Ny-1; y++)
    	  for (x=1; x<Nx-1; x++)
              V_NEW[x][y] = -0.25 * ro[y][x] * h * h +
         		     0.25 * (V[y-1][x] + V[y+1][x] + V[y][x-1] + V[y][x+1]);

//roznica
     roznica = 0;
     for (y=1; y<Ny-1; y++)
     	 for (x=1; x<Nx-1; x++)
             roznica += (V_NEW[y][x] - V[y][x]) * (V_NEW[y][x] - V[y][x]);

     roznica = sqrt(roznica);
     
     if (roznica < epsilon) break;

//przepisz
     for (y=1; y<Ny-1; y++)
     	 for (x=1; x<Nx-1; x++)
             V[y][x] = V_NEW[y][x];

 }
}

void wypisz_V_ro (double** V, double** ro, int Nx, int Ny)
{ 
 int x, y;
 
 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++)
	 printf ("%d %d %f %f\n", x, y, V[y][x], ro[y][x]);
}

int main()
{
 int Nx, Ny, k, kMAX;
 double h, epsilon;
 double ** ro;
 double ** V;
 double ** V_NEW;

 Nx = Ny = 1000;
 h = 0.001;
 kMAX = 20;
 epsilon = 0.000001;

 ro = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     ro[k] = (double*) malloc(Nx * sizeof(double));

 V = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     V[k] = (double*) malloc(Nx * sizeof(double)); 

 V_NEW = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     V_NEW[k] = (double*) malloc(Nx * sizeof(double));

 ewolucja_V(V_NEW, V, ro, Nx, Ny, h, kMAX, epsilon);

// wypisz_V_ro (V_NEW, ro, Nx, Ny);

 for (k=0; k<Ny; k++)
     free(ro[k]);
 free(ro);

 for (k=0; k<Ny; k++)
     free(V[k]);
 free(V);

 for (k=0; k<Ny; k++)
     free(V_NEW[k]);
 free(V_NEW);

 return 0;
}
