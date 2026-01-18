#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

// Последовательная сортировка выбором
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;
        std::swap(arr[i], arr[min_idx]);
    }
}

// Параллельная сортировка выбором (внешний цикл распараллелить)
void selectionSortParallel(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;

        // поиск минимального элемента параллельно
        #pragma omp parallel
        {
            int local_min = min_idx;
            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[local_min]) local_min = j;
            }
            #pragma omp critical
            {
                if (arr[local_min] < arr[min_idx]) min_idx = local_min;
            }
        }

        std::swap(arr[i], arr[min_idx]);
    }
}

int main() {
    std::vector<int> sizes = {1000, 10000};

    for (int n : sizes) {
        std::cout << "\n=== Array size: " << n << " ===\n";

        std::vector<int> arr(n);
        std::mt19937 gen(42);
        std::uniform_int_distribution<> dis(0, 1000000);
        for (int &x : arr) x = dis(gen);

        // ---------- Sequential ----------
        auto arr_seq = arr;
        auto start_seq = std::chrono::high_resolution_clock::now();
        selectionSort(arr_seq);
        auto end_seq = std::chrono::high_resolution_clock::now();
        double seq_time = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
        std::cout << "Sequential Selection Sort Time: " << seq_time << " ms\n";

        // ---------- Parallel ----------
        auto arr_par = arr;
        auto start_par = std::chrono::high_resolution_clock::now();
        selectionSortParallel(arr_par);
        auto end_par = std::chrono::high_resolution_clock::now();
        double par_time = std::chrono::duration<double, std::milli>(end_par - start_par).count();
        std::cout << "Parallel Selection Sort Time: " << par_time << " ms\n";
    }

    return 0;
}
