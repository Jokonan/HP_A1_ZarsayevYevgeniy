#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>

// -----------------------------
// Вспомогательные функции
// -----------------------------

// Генерация вектора случайных чисел в диапазоне [minVal, maxVal]
std::vector<int> generateArray(size_t size, int minVal, int maxVal) {
    std::vector<int> arr(size);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(minVal, maxVal);

    for (auto& x : arr)
        x = dist(gen);

    return arr;
}

// Удобный алиас для времени
using Clock = std::chrono::high_resolution_clock;

// -----------------------------
// Задание 1
// -----------------------------
void task1() {
    std::cout << "\n--- Task 1 ---\n";

    const size_t SIZE = 50'000;

    // Динамическое выделение памяти
    int* arr = new int[SIZE];

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100);

    long long sum = 0;

    // Заполнение и подсчёт суммы
    for (size_t i = 0; i < SIZE; ++i) {
        arr[i] = dist(gen);
        sum += arr[i];
    }

    double average = static_cast<double>(sum) / SIZE;
    std::cout << "Average value: " << average << '\n';

    // Корректное освобождение памяти
    delete[] arr;
}

// -----------------------------
// Задание 2
// -----------------------------
void task2(const std::vector<int>& arr, int& minVal, int& maxVal, double& timeMs) {
    auto start = Clock::now();

    minVal = std::numeric_limits<int>::max();
    maxVal = std::numeric_limits<int>::min();

    // Последовательный поиск min и max
    for (int x : arr) {
        if (x < minVal) minVal = x;
        if (x > maxVal) maxVal = x;
    }

    auto end = Clock::now();
    timeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// -----------------------------
// Задание 3
// -----------------------------
void task3(const std::vector<int>& arr, int& minVal, int& maxVal, double& timeMs) {
    auto start = Clock::now();

    minVal = std::numeric_limits<int>::max();
    maxVal = std::numeric_limits<int>::min();

    // Параллельный поиск min и max
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
    }

    auto end = Clock::now();
    timeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

// -----------------------------
// Задание 4
// -----------------------------
void task4() {
    std::cout << "\n--- Task 4 ---\n";

    const size_t SIZE = 5'000'000;
    auto arr = generateArray(SIZE, 1, 100);

    // Последовательное вычисление среднего
    auto startSeq = Clock::now();
    long long sumSeq = 0;

    for (int x : arr)
        sumSeq += x;

    double avgSeq = static_cast<double>(sumSeq) / SIZE;
    auto endSeq = Clock::now();

    double timeSeq = std::chrono::duration<double, std::milli>(endSeq - startSeq).count();

    // Параллельное вычисление среднего (OpenMP reduction)
    auto startPar = Clock::now();
    long long sumPar = 0;

#pragma omp parallel for reduction(+:sumPar)
    for (int i = 0; i < static_cast<int>(SIZE); ++i)
        sumPar += arr[i];

    double avgPar = static_cast<double>(sumPar) / SIZE;
    auto endPar = Clock::now();

    double timePar = std::chrono::duration<double, std::milli>(endPar - startPar).count();

    // Вывод результатов
    std::cout << "Sequential average: " << avgSeq << ", time: " << timeSeq << " ms\n";
    std::cout << "Parallel average:   " << avgPar << ", time: " << timePar << " ms\n";
}

// -----------------------------
// main
// -----------------------------
int main() {
    // Задание 1
    task1();

    // Подготовка массива для заданий 2 и 3
    const size_t SIZE = 1'000'000;
    auto arr = generateArray(SIZE, 1, 100);

    // Задание 2
    std::cout << "\n--- Task 2 ---\n";
    int minSeq, maxSeq;
    double timeSeq;

    task2(arr, minSeq, maxSeq, timeSeq);
    std::cout << "Min: " << minSeq << ", Max: " << maxSeq
              << ", Time: " << timeSeq << " ms\n";

    // Задание 3
    std::cout << "\n--- Task 3 ---\n";
    int minPar, maxPar;
    double timePar;

    task3(arr, minPar, maxPar, timePar);
    std::cout << "Min: " << minPar << ", Max: " << maxPar
              << ", Time: " << timePar << " ms\n";

    // Задание 4
    task4();

    return 0;
}
