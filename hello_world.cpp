#include <iostream>

using namespace std;

int main() {
    const int liczba_elementow = 4096;
    int tablica[liczba_elementow];
    cout << "Hello world!" << endl;
    for (int i = 0; i < liczba_elementow; i++) {
	tablica[i] = i;
    }

    for (int i = 0; i < liczba_elementow; i++) {
	tablica[i] = 2 * tablica[i];
    }

    for (int i = 0; i < 10; i++) {
	cout << "indeks " << i << " wartosc " << tablica[i] << endl;
    }
    cout << "indeks 4095 wartosc " << tablica[4095] << endl;
    return 0;
}
