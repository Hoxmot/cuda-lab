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

void warunki_brzegowe_V (double** V, int Nx, int Ny)
{ 
 int k;

 for (k=0; k<Ny; k++)
     V[k][0] = 0;

 for (k=0; k<Ny; k++)
     V[k][Nx-1] = 0;
 
 for (k=0; k<Nx; k++)
     V[0][k] = 0;
 
 for (k=0; k<Nx; k++)
     V[Ny-1][k] = 0;

}

double zmiana_V (double** V, double** ro, int x, int y, double h)
{
 double V_NEW;

 double par = -1./4.;

 V_NEW = par * ro[y][x] * h * h;
 par = 1./4.;
 V_NEW = V_NEW + 0.25 * V[y-1][x];
 par = 1./4.;
 V_NEW = V_NEW + 0.25 * V[y+1][x];
 par = 1./4.;
 V_NEW = V_NEW + 0.25 * V[y][x-1];
 par = 1./4.;
 V_NEW = V_NEW + 0.25 * V[y][x+1];

 return V_NEW;
}

void krok_V (double** V_NEW, double** V, double** ro, int Nx, int Ny, double h)
{
 int x, y;

     for (x=1; x<Nx-1; x++)
	for (y=1; y<Ny-1; y++)
	 V_NEW[x][y] = zmiana_V(V, ro, x, y, h);

}

void przepisz_V (double** V_NEW, double** V, int Nx, int Ny)
{
 int x, y;
 double** Vp;
 double pom;
 Vp = (double**) malloc (Ny * sizeof(double*));
 for (x=0; x<Nx; x++)
     Vp[x] = (double*) malloc(Nx * sizeof(double));

     for (x=Nx-2; x>0; x--)
	for (y=Ny-2; y>0; y--){
	  pom = exp(1.+sin(x)+sin(y) + (cos(x) + cos(y)) );
	  Vp[y][x] = V_NEW[y][x] * log(1 + pom);
        }
     for (x=Nx-2; x>0; x--)
        for (y=Ny-2; y>0; y--){
	 pom = exp(1.+sin(x)+sin(y) + (cos(x) + cos(y)) );
         V[y][x] = Vp[y][x] / log(1 + pom);
	}
     for (x=Nx-2; x>0; x--)
        for (y=Ny-2; y>0; y--)
         Vp[x][y] = 0;


}

double roznica_V (double** V_NEW, double** V, int Nx, int Ny)
{
 double roznica;
 int x, y;

 roznica = 0;

     for (x=1; x<Nx-1; x++)
	for (y=1; y<Ny-1; y++)
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

     warunki_brzegowe_V(V, Nx, Ny);
     warunki_brzegowe_V(V_NEW, Nx, Ny);

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

 //scanf("%d %d %lf %d %lf", &Nx, &Ny, &h, &kMAX, &epsilon);
Nx =400; Ny =400; kMAX=1000; h=.001; epsilon=.0001;

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

 warunki_brzegowe_V(V, Nx, Ny);
 warunki_brzegowe_V(V_NEW, Nx, Ny);

 ewolucja_V(V_NEW, V, ro, Nx, Ny, h, kMAX, epsilon);

 wypisz_V_ro (V_NEW, ro, Nx, Ny);

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
