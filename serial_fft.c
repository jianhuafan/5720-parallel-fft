/**  FFT using Cooley–Tukey FFT algorithm
 * 
 */ 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct Complex {
    // a + bi
    double a;
    double b;
} Complex;

Complex comp_create(double a, double b) {
    Complex ret;
    ret.a = a;
    ret.b = b;
    return ret;
}

Complex comp_add(Complex c1, Complex c2) {
    Complex ret = c1;
    ret.a += c2.a;
    ret.b += c2.b;
    return ret;
}

Complex comp_sub(Complex c1, Complex c2) {
    Complex ret = c1;
    ret.a -= c2.a;
    ret.b -= c2.b;
    return ret;
}

Complex comp_mul(Complex c1, Complex c2) {
    Complex ret;
    ret.a = c1.a * c2.a - c1.b * c2.b;
    ret.b = c1.b * c2.a + c1.a * c2.b;
    return ret;
}

Complex comp_euler(double x) {
    Complex ret;
    ret.a = cos(x);
    ret.b = sin(x);
    return ret;
}

void comp_print(Complex comp) {
    if (comp.b < 0) {
        printf("%.6f - %.6f i\n", comp.a, -comp.b);
    } else {
        printf("%.6f + %.6f i\n", comp.a, comp.b);
    }
}

void comp_mul_self(Complex *c1, Complex *c2) {
    double c1a = c1->a;
    c1->a = c1a * c2->a - c1->b * c2->b;
    c1->b = c1->b * c2->a + c1a * c2->b;
}

void fft(const Complex *in, Complex *out, int step, int n) {
    int i;
    int half_n = n >> 1;
    const double PI = acos(-1);
    Complex ep = comp_euler(-PI / (double)half_n);
    Complex ei;
    Complex *ep_ptr = &ep;
    Complex *ei_ptr = &ei;
    if (!half_n) {
        *out = *in;
    } else {
        fft(in, out, step << 1, half_n);
        fft(in + step, out + half_n, step << 1, half_n);
        ei_ptr->a = 1;
        ei_ptr->b = 0;
        for (i = 0; i < half_n; i++) {
            Complex even = out[i];
            Complex *even_ptr = out + i;
            Complex *odd_ptr = even_ptr + half_n;
            comp_mul_self(odd_ptr, ei_ptr);
            even_ptr->a += odd_ptr->a;
            even_ptr->b += odd_ptr->b;
            odd_ptr->a = even.a - odd_ptr->a;
            odd_ptr->b = even.b - odd_ptr->b;
            comp_mul_self(ei_ptr, ep_ptr);
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        return 0;
    }
    int n;
    int i;
    n = atoi(argv[1]);
    Complex *in, *out;
    in = (Complex *) malloc(sizeof(Complex) * (size_t)n);
    out = (Complex *) malloc(sizeof(Complex) * (size_t)n);
    for (i = 0; i < n; i++) {
        in[i].a = rand() % 10;
        in[i].b = 0;
    }
    puts("#### Original Signal ####");
    for (i = 0; i < n; i++) {
        comp_print(in[i]);
    }
    fft(in ,out, 1, n);
    puts("#### Fourier Transform Result ####");
    for (i = 0; i < n; i++) {
        comp_print(out[i]);
    }
}