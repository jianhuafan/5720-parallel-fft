/**  FFT using Cooley–Tukey FFT algorithm
 *  flag: -n size of sequence, r resursive, i iterative
 */ 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <sys/times.h>
#include <sys/types.h>
#include <omp.h>

#define BILLION 1000000000L
#define FFT_FORWARD -1
#define FFT_BACKWARD 1

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

void resursive_fft(const Complex *in, Complex *out, int step, int n) {
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
        resursive_fft(in, out, step << 1, half_n);
        resursive_fft(in + step, out + half_n, step << 1, half_n);
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

unsigned int bit_reverse(unsigned int num, unsigned int bits);
void bit_reverse_array(Complex *in, Complex *out, int n);

void iterative_fft(Complex *in, Complex *out, int n) {
    int i, s, j, k;
    const double PI = acos(-1);
    bit_reverse_array(in, out, n);
    Complex ei;
    Complex *ei_ptr = &ei;
    for (s = 1; s < log2(n) + 1; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        Complex ep = comp_euler(-PI / (double)m2);
        Complex *ep_ptr = &ep; 
        ei_ptr->a = 1;
        ei_ptr->b = 0;
        for (j = 0; j < m2; j++) {
            for (k = j; k < n; k += m) {
                Complex t = comp_mul(ei, out[k + m2]);
                Complex u = out[k];
                Complex *even_ptr = out + k;
                Complex *odd_ptr = out + k + m2;
                even_ptr->a = u.a + t.a;
                even_ptr->b = u.b + t.b;
                odd_ptr->a = u.a - t.a;
                odd_ptr->b = u.b - t.b;
            }
            comp_mul_self(ei_ptr, ep_ptr);
        }
    }
}

void init_w_table(Complex *W, int n, int sign) {
    int i;
    const double PI = acos(-1);
    W[0] = comp_create(1, 0);
    W[1] = comp_euler(sign * PI / (double)(n / 2));
    #pragma omp parallel shared ( W ) private ( i )
    {
        #pragma omp for nowait
        for (i = 2; i < n / 2; i++) {
            W[i] = comp_create(1, 0);
            int j;
            for (j = 0; j < i; j++) {
                comp_mul_self(&W[i], &W[1]);
            }
        }
    }
}

void openmp_1d_fft(Complex *in, Complex *out, int n, int sign) {
    unsigned long step = 1;
    unsigned long i;
    unsigned long a = n / 2;
    unsigned long j;
    const double PI = acos(-1);
    bit_reverse_array(in, out, n);
    Complex *W;
    W = (Complex *) malloc(sizeof(Complex) * (size_t)a);
    init_w_table(W, n, sign);
    unsigned long size = log2(n);
    for (j = 0; j < size; j++) {
        #pragma omp parallel shared(j, in, out, W, step, a, n) private(i)
        {
            #pragma omp for
            for (i = 0; i < n; i++) {
                if (!(i & step)) {
                    Complex u = out[i];
                    Complex t = comp_mul(W[(i * a) % (step * a)], out[i + step]);
                    Complex *even_ptr = out + i;
                    Complex *odd_ptr = out + i + step;
                    even_ptr->a = u.a + t.a;
                    even_ptr->b = u.b + t.b;
                    odd_ptr->a = u.a - t.a;
                    odd_ptr->b = u.b - t.b;
                }
            }
        }
        step = step * 2;
        a = a / 2;
    }
}

void openmp_2d_fft(Complex *in, Complex *out, int m, int n, int sign) {
    int i, j;
    #pragma omp parallel
    {
        #pragma omp for private(i, j)
        for (i = 0; i < m; i++) {
            openmp_1d_fft(in + i * n, out + i * n, n, sign);
        }
        Complex *temp_out = (Complex*) malloc(sizeof(Complex)* m * n);
        for (i = 0; i < n; i++) {
            Complex *temp = (Complex*) malloc(sizeof(Complex)* m);
            for (j = 0; j < m; j++) {
                temp[j] = out[j * n + i];
            }
            openmp_1d_fft(temp, temp_out + i * m, m, sign);
        }
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                out[i * n + j] = temp[j * m + i];
            }
        }
    }
}

void bit_reverse_array(Complex *in, Complex *out, int n) {
    unsigned int i;
    unsigned int bits = log2(n);
    #pragma omp parallel shared ( in, out, n, bits ) private ( i )
    {
        #pragma omp for nowait
        for (i = 0; i < n; i++) {
            int reversed_i = bit_reverse(i, bits);
            Complex *out_ptr = out + reversed_i;
            Complex *in_ptr = in + i;
            out_ptr->a = in_ptr->a;
            out_ptr->b = in_ptr->b;
        }
    }
}

unsigned int bit_reverse(unsigned int num, unsigned int bits) {
    unsigned int reversed_num = 0;
    unsigned int i;
    for (i = 0; i < bits; i++) {
        reversed_num <<= 1;
        reversed_num |= (num & 1);
        num >>= 1;
    }
    return reversed_num;
}