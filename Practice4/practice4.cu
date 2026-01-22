#include <iostream>          // ввод и вывод в консоль
#include <cuda_runtime.h>    // основные функции CUDA
#include <cstdlib>           // rand(), srand()
#include <ctime>             // time() для генерации случайных чисел

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at line " << __LINE__ << std::endl;   \
        exit(1);                                              \
    }                                                         \
} while (0)

// Каждый поток суммирует часть массива
// Используется только глобальная память
__global__ void reduceGlobal(const float* arr, float* result, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    float localSum = 0.0f;                            // локальная сумма потока

    // Поток проходит по массиву с шагом, равным количеству всех потоков
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        localSum += arr[i];                          // добавляем элемент массива
    }

    // Безопасно добавляем сумму потока в общий результат
    atomicAdd(result, localSum);
}

// Редукция с использованием разделяемой памяти
__global__ void reduceShared(const float* arr, float* result, int n) {

    extern __shared__ float sdata[]; // разделяемая память блока

    int tid = threadIdx.x;           // индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid; // глобальный индекс потока

    float sum = 0.0f;                // локальная сумма потока

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += arr[i];               // суммируем элементы массива
    }

    sdata[tid] = sum;                // сохраняем сумму в shared memory
    __syncthreads();                 // ждём, пока все потоки запишут данные

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            sdata[tid] += sdata[tid + step]; // объединяем суммы
        }
        __syncthreads();             // синхронизация после каждого шага
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]); // добавляем сумму блока в результат
    }
}

// Сортировка пузырьком для небольших подмассивов
__global__ void bubbleSortSubarrays(float* arr, int n, int chunkSize) {

    int blockId = blockIdx.x;               // номер блока
    int start = blockId * chunkSize;        // начало подмассива
    int end = min(start + chunkSize, n);    // конец подмассива

    for (int i = start; i < end; ++i) {     // внешний цикл пузырьковой сортировки
        for (int j = start; j < end - 1; ++j) { // внутренний цикл
            if (arr[j] > arr[j + 1]) {      // сравниваем соседние элементы
                float tmp = arr[j];         // временная переменная
                arr[j] = arr[j + 1];        // меняем элементы местами
                arr[j + 1] = tmp;
            }
        }
    }
}

int main() {

    const int sizes[3] = {10000, 100000, 1000000}; // размеры массивов

    for (int s = 0; s < 3; ++s) {

        int N = sizes[s];                   // текущий размер массива
        std::cout << "\nArray size: " << N << std::endl;

        float* h_arr = new float[N];        // массив на CPU
        srand(time(nullptr));               // инициализация генератора случайных чисел

        for (int i = 0; i < N; ++i) {
            h_arr[i] = (float)rand() / RAND_MAX; // случайные числа от 0 до 1
        }

        float* d_arr;                       // массив на GPU
        float* d_result;                    // результат редукции на GPU

        CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float))); // выделяем память на GPU
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float),
                               cudaMemcpyHostToDevice));   // копируем данные на GPU

        cudaEvent_t start, stop;            // таймеры CUDA
        float timeMs;                       // время выполнения

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float))); // обнуляем результат

        CUDA_CHECK(cudaEventRecord(start)); // старт замера времени
        reduceGlobal<<<256, 256>>>(d_arr, d_result, N); // запуск ядра
        CUDA_CHECK(cudaEventRecord(stop));  // конец замера
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));

        std::cout << "Reduction (global): " << timeMs << " ms" << std::endl;

        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float))); // обнуляем результат

        CUDA_CHECK(cudaEventRecord(start));
        reduceShared<<<256, 256, 256 * sizeof(float)>>>(d_arr, d_result, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));

        std::cout << "Reduction (shared): " << timeMs << " ms" << std::endl;

        int chunkSize = 256;                                // размер подмассива
        int blocks = (N + chunkSize - 1) / chunkSize;      // количество блоков

        CUDA_CHECK(cudaEventRecord(start));
        bubbleSortSubarrays<<<blocks, 1>>>(d_arr, N, chunkSize);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));

        std::cout << "Bubble sort: " << timeMs << " ms" << std::endl;

        CUDA_CHECK(cudaFree(d_arr));        // освобождаем память GPU
        CUDA_CHECK(cudaFree(d_result));
        delete[] h_arr;                     // освобождаем память CPU
    }

    return 0;                               // завершение программы
}
