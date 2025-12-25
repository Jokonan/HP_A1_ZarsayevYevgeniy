#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;
using namespace chrono;

// ================== ВСПОМОГАТЕЛЬНЫЕ ==================

void fillArray(int* arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand();
}

void copyArray(int* src, int* dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

// ================== ПОСЛЕДОВАТЕЛЬНЫЕ СОРТИРОВКИ ==================

// Пузырёк
void bubbleSort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

// Выбор
void selectionSort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[minIdx])
                minIdx = j;
        swap(arr[i], arr[minIdx]);
    }
}

// Вставки
void insertionSort(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// ================== ПАРАЛЛЕЛЬНЫЕ СОРТИРОВКИ ==================

// Параллельный пузырёк (odd-even sort — корректный вариант)
void bubbleSortOMP(int* arr, int n) {
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            #pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
        } else {
            #pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2)
                if (arr[i] > arr[i + 1])
                    swap(arr[i], arr[i + 1]);
        }
    }
}

// Параллельная сортировка выбором (корректно, но без ускорения)
void selectionSortOMP(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;

        #pragma omp parallel
        {
            int localMin = minIdx;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++)
                if (arr[j] < arr[localMin])
                    localMin = j;

            #pragma omp critical
            {
                if (arr[localMin] < arr[minIdx])
                    minIdx = localMin;
            }
        }
        swap(arr[i], arr[minIdx]);
    }
}

// Параллельные вставки (через ordered — корректно, но без ускорения)
void insertionSortOMP(int* arr, int n) {
    #pragma omp parallel for ordered
    for (int i = 1; i < n; i++) {
        #pragma omp ordered
        {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// ================== ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ==================

void testAll(int n) {
    cout << "\nРазмер массива: " << n << endl;

    int* original = new int[n];
    int* arr = new int[n];

    fillArray(original, n);

    auto run = [&](const char* name, void (*sortFunc)(int*, int)) {
        copyArray(original, arr, n);
        auto start = high_resolution_clock::now();
        sortFunc(arr, n);
        auto end = high_resolution_clock::now();
        cout << name << ": "
             << duration_cast<milliseconds>(end - start).count()
             << " ms\n";
    };

    run("Bubble seq", bubbleSort);
    run("Bubble omp", bubbleSortOMP);

    run("Selection seq", selectionSort);
    run("Selection omp", selectionSortOMP);

    run("Insertion seq", insertionSort);
    run("Insertion omp", insertionSortOMP);

    delete[] original;
    delete[] arr;
}

// ================== MAIN ==================

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    srand(time(nullptr));

    testAll(1000);
    testAll(10000);
    testAll(100000);

    return 0;
}
