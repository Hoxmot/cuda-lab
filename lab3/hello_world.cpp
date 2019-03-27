#include <iostream>

using namespace std;

int main() {
    const int elem_number = 4096;
    int tab[elem_number];
    cout << "Hello world!" << endl;
    for (int i = 0; i < elem_number; i++) {
	tab[i] = i;
    }

    for (int i = 0; i < elem_number; i++) {
	tab[i] = 2 * tab[i];
    }

    for (int i = 0; i < 10; i++) {
	cout << "Index " << i << " value " << tab[i] << endl;
    }
    cout << "Index 4095 value " << tab[4095] << endl;
    return 0;
}
