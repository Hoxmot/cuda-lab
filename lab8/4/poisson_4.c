#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double funkcja_ro (int x, int y, double h)
{
 double pom;

 if (x==y && x==15)  pom = 1;
 else pom = 0;

 return pom;
}

double funkcja_V (int x, int y, double h)
{
 double pom;

 pom = sin(h*x)*sin(h*y);

 return pom;
}


void ustal_gestosc (double** ro, int Nx, int Ny, double h)
{
 int x, y;
 
 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++)
	 ro[y][x] = funkcja_ro(x,y,h);
}

void poczatkowe_V (double** V, int Nx, int Ny, double h)
{
 int x, y;

 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++)
         V[y][x] = funkcja_V(x,y,h);

}

double V_brzegi (double** ro, int Nx, int Ny, double h, int X, int Y)
{
 int x, y;
 double wynik = 0;

 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++){
	 wynik += ro[x][y] * 1. / sqrt( (x-X)*(x-X) + (y-Y)*(y-Y) );
     }
 return wynik;
}

void warunki_brzegowe_V (double** V, double** ro, int Nx, int Ny, double h)
{ 
 int k;

 for (k=0; k<Ny; k++)
     V[k][0] = V_brzegi(ro, Nx, Ny, h, k, 0);

 for (k=0; k<Ny; k++)
     V[k][Nx-1] = V_brzegi(ro, Nx, Ny, h, k, Nx-1); 
 
 for (k=0; k<Nx; k++)
     V[0][k] = V_brzegi(ro, Nx, Ny, h, 0, k);
 
 for (k=0; k<Nx; k++)
     V[Ny-1][k] = V_brzegi(ro, Nx, Ny, h, Ny-1, k);
}

double zmiana_V (double** V, double** ro, int x, int y, double h)
{
 double V_NEW;

 V_NEW = -.25 * ro[y][x] * h * h +
	 0.25 * (V[y-1][x] + V[y+1][x] + V[y][x-1] + V[y][x+1]);

 return V_NEW;
}

void krok_V (double** V_NEW, double** V, double** ro, int Nx, int Ny, double h)
{
 int x, y;

 for (y=1; y<Ny-1; y++)
     for (x=1; x<Nx-1; x++)
	 V_NEW[x][y] = zmiana_V(V, ro, x, y, h);

}

void przepisz_V (double** V_NEW, double** V, int Nx, int Ny)
{
 int x, y;
 for (y=1; y<Ny-1; y++)
     for (x=1; x<Nx-1; x++)
	 V[y][x] = V_NEW[y][x];
}

void przepisz_V_all (double** V_NEW, double** V, int Nx, int Ny)
{
 int x, y;
 for (y=0; y<Ny; y++)
     for (x=0; x<Nx; x++)
         V[y][x] = V_NEW[y][x];
}


double roznica_V (double** V_NEW, double** V, int Nx, int Ny)
{
 double roznica;
 int x, y;

 roznica = 0;

 for (y=1; y<Ny-1; y++)
     for (x=1; x<Nx-1; x++)
	 roznica += (V_NEW[y][x] - V[y][x]) * (V_NEW[y][x] - V[y][x]);

 roznica = sqrt(roznica);

 return roznica;
} 

void ewolucja_V (double** V_NEW, double** V, double** ro, int Nx, int Ny, double h, int kMAX, double epsilon)
{ 
 int k;
 double roznica;
 for (k=0; k<kMAX; k++) {
     krok_V(V_NEW, V, ro, Nx, Ny, h);
     roznica = roznica_V(V_NEW, V, Nx, Ny);
     if (roznica < epsilon) break;
     przepisz_V(V_NEW, V, Nx, Ny);
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

 Nx = Ny = 300;
 h = 0.001;
 kMAX = 10;
 epsilon = 0.0001;

 ro = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     ro[k] = (double*) malloc(Nx * sizeof(double));

 V = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     V[k] = (double*) malloc(Nx * sizeof(double)); 

 V_NEW = (double**) malloc (Ny * sizeof(double*));
 for (k=0; k<Nx; k++)
     V_NEW[k] = (double*) malloc(Nx * sizeof(double));

 ustal_gestosc(ro,  Nx, Ny, h);

 poczatkowe_V(V, Nx, Ny, h);

 warunki_brzegowe_V(V, ro, Nx, Ny, h);
 przepisz_V_all (V_NEW, V, Nx, Ny);

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
