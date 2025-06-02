// test_runner.cpp

#include <iostream>

// ✅ 선언이 필요
extern "C" void vec_add_launcher(float* a, float* b, float* c, int N);

int main() {
    const int N = 100;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    vec_add_launcher(a, b, c, N);

    for (int i = 0; i < 5; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    return 0;
}
