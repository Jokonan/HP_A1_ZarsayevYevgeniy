#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <omp.h>

int main() {
    const int N = 10000;
    std::vector<int> arr(N);

    // Генерация случайных чисел
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1000000);
    for (int &x : arr) x = dis(gen);

    // ================= Последовательный поиск min/max =================
    auto start_seq = std::chrono::high_resolution_clock::now();
    int min_val = std::numeric_limits<int>::max();
    int max_val = std::numeric_limits<int>::min();
    for (int x : arr) {
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();

    std::cout << "Sequential Min: " << min_val << ", Max: " << max_val 
              << ", Time: " << seq_time << " ms\n";

    // ================= Параллельный поиск min/max =================
    min_val = std::numeric_limits<int>::max();
    max_val = std::numeric_limits<int>::min();

    auto start_par = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < N; i++) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    double par_time = std::chrono::duration<double, std::milli>(end_par - start_par).count();

    std::cout << "Parallel Min: " << min_val << ", Max: " << max_val 
              << ", Time: " << par_time << " ms\n";

    return 0;
}
