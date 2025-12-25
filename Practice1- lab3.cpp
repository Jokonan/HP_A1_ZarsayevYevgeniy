#include <iostream>
#include <cstdlib>   // для rand() и srand()
#include <ctime>     // для time()
#include <omp.h>     // для OpenMP
#ifdef _WIN32
#include <windows.h>
#endif


using namespace std;

// Функция для подсчёта среднего значения (параллельно)
double calculateAverage(int* arr, int size) {
    double sum = 0.0;

    // Параллельно считаем сумму элементов массива
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return sum / size;
}

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif


    int size;

    cout << "Введите размер массива: ";
    cin >> size;

    // Проверка на корректный размер
    if (size <= 0) {
        cout << "Размер массива должен быть больше 0" << endl;
        return 1;
    }

    // Инициализируем генератор случайных чисел
    srand(time(nullptr));

    // 1. Выделяем динамическую память под массив
    int* arr = new int[size];

    // Заполняем массив случайными числами
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100; // числа от 0 до 99
    }

    // Вывод массива (для проверки)
    cout << "Массив: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // 2–3. Считаем среднее значение (параллельно через OpenMP)
    double average = calculateAverage(arr, size);

    cout << "Среднее значение массива: " << average << endl;

    // 4. Освобождаем динамическую память
    delete[] arr;
    arr = nullptr;

    return 0;
}
