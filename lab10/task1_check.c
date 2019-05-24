#include <stdio.h>

#define LEN 32

int main() {
    double res = 0;
    for (int i = 0; i < LEN; i++) {
        res += (i + 1) / 42.;
        printf("%d, %f\n", i+1, (i+1)/42.);
    }
    printf("%f\n", res);
    return 0;
}