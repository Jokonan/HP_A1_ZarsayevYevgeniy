#include <iostream>              // вывод в консоль
#include <cuda_runtime.h>        // CUDA функции
#include <cstdlib>               // rand()
#include <ctime>                 // time()
#include <chrono>                // таймер CPU

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(call)                                  \
do {                                                      \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        std::cerr << "CUDA error: "                       \
                  << cudaGetErrorString(err)              \
                  << " at line " << __LINE__ << std::endl;\
        exit(1);                                           \
    }                                                     \
} while (0)



// ЗАДАНИЕ 1 - Сумма массива на GPU с использованием глобальной памяти

__global__ void sumGlobal(const float* arr, float* result, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // индекс потока
    float localSum = 0.0f;                              // локальная сумма

    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        localSum += arr[i];                             // суммируем элементы
    }

    atomicAdd(result, localSum);                        // добавляем в результат
}

float cpuSum(const float* arr, int n) {

    float sum = 0.0f;                                   // итоговая сумма

    for (int i = 0; i < n; ++i) {
        sum += arr[i];                                  // последовательная сумма
    }

    return sum;                                         // возвращаем результат
}



// ЗАДАНИЕ 2 - Префиксная сумма с использованием shared memory

__global__ void prefixScan(float* data, int n) {

    extern __shared__ float sdata[];                     // shared memory блока

    int tid = threadIdx.x;                               // индекс в блоке
    int idx = blockIdx.x * blockDim.x + tid;             // глобальный индекс

    if (idx < n)
        sdata[tid] = data[idx];                          // копируем элемент
    else
        sdata[tid] = 0.0f;                               // заполняем нулями

    __syncthreads();                                     // ждём все потоки

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;                                // временное значение
        if (tid >= offset)
            val = sdata[tid - offset];                   // берём прошлое значение
        __syncthreads();
        sdata[tid] += val;                               // накапливаем сумму
        __syncthreads();
    }

    if (idx < n)
        data[idx] = sdata[tid];                          // записываем результат
}

void cpuPrefixScan(float* arr, int n) {

    for (int i = 1; i < n; ++i) {
        arr[i] += arr[i - 1];                            // префиксная сумма
    }
}



// ЗАДАНИЕ 3 - Гибридная обработка - CPU, GPU, Гибридный подход

// CPU обработка всего массива
void cpuProcessFull(float* arr, int n) {

    for (int i = 0; i < n; ++i) {
        arr[i] *= 2.0f;                                  // простая операция
    }
}

// GPU обработка всего массива
__global__ void gpuProcessFull(float* arr, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // индекс потока

    if (idx < n) {
        arr[idx] *= 2.0f;                                // умножаем элемент
    }
}

// GPU обработка половины массива
__global__ void gpuProcessHalf(float* arr, int offset, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // индекс потока

    if (idx + offset < n) {
        arr[idx + offset] *= 2.0f;                       // вторая половина
    }
}

// CPU обработка части массива
void cpuProcessPart(float* arr, int start, int end) {

    for (int i = start; i < end; ++i) {
        arr[i] *= 2.0f;                                  // CPU часть массива
    }
}


// =======================================================
// MAIN
// =======================================================

int main() {

    srand(time(nullptr));                                // инициализация rand

    // ---------------- ЗАДАНИЕ 1 ----------------
    {
        int N = 100000;                                  // размер массива
        std::cout << "\nTASK 1: Sum of array\n";

        float* h_arr = new float[N];                     // массив CPU
        for (int i = 0; i < N; ++i)
            h_arr[i] = 1.0f;                              // заполнение единицами

        auto cpuStart = std::chrono::high_resolution_clock::now();
        float cpuResult = cpuSum(h_arr, N);              // CPU сумма
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        std::cout << "CPU sum: " << cpuResult << std::endl;
        std::cout << "CPU time: "
                  << std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count()
                  << " ms\n";

        float *d_arr, *d_result;
        CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float)));   // память GPU
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));    // результат

        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float),
                               cudaMemcpyHostToDevice));    // копирование
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));  // обнуление

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);                           // старт GPU таймера
        sumGlobal<<<256, 256>>>(d_arr, d_result, N);      // запуск ядра
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);                       // ожидание

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);      // время GPU

        float gpuResult;
        CUDA_CHECK(cudaMemcpy(&gpuResult, d_result,
                               sizeof(float),
                               cudaMemcpyDeviceToHost)); // копия результата

        std::cout << "GPU sum: " << gpuResult << std::endl;
        std::cout << "GPU time: " << gpuTime << " ms\n";

        delete[] h_arr;                                   // очистка CPU
        cudaFree(d_arr);
        cudaFree(d_result);
    }

    // ---------------- ЗАДАНИЕ 2 ----------------
    {
        int N = 1024 * 1024;                              // размер массива
        std::cout << "\nTASK 2: Prefix sum\n";

        float* h_arr = new float[N];                      // массив CPU
        for (int i = 0; i < N; ++i)
            h_arr[i] = 1.0f;                              // единицы

        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuPrefixScan(h_arr, N);                          // CPU scan
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        std::cout << "CPU scan time: "
                  << std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count()
                  << " ms\n";

        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float))); // память GPU
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float),
                               cudaMemcpyHostToDevice));  // копирование

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        prefixScan<<<N / 1024, 1024, 1024 * sizeof(float)>>>(d_arr, N); // GPU scan
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);      // время GPU

        std::cout << "GPU scan time: " << gpuTime << " ms\n";

        delete[] h_arr;
        cudaFree(d_arr);
    }

    // ---------------- ЗАДАНИЕ 3 ----------------
    {
        int N = 1000000;                                  // размер массива
        std::cout << "\nTASK 3: CPU vs GPU vs Hybrid\n";

        float* h_arr = new float[N];                      // массив CPU
        for (int i = 0; i < N; ++i)
            h_arr[i] = 1.0f;                              // единицы

        float* d_arr;
        CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float))); // память GPU

        // ---------- CPU ----------
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuProcessFull(h_arr, N);                         // CPU обработка
        auto cpuEnd = std::chrono::high_resolution_clock::now();

        std::cout << "CPU time: "
                  << std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count()   //Вывод времени
                  << " ms\n";

        // ---------- GPU ----------
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float),
                               cudaMemcpyHostToDevice)); // копирование

        cudaEvent_t gStart, gStop;
        cudaEventCreate(&gStart);
        cudaEventCreate(&gStop);

        cudaEventRecord(gStart);
        gpuProcessFull<<<(N + 255) / 256, 256>>>(d_arr, N); // GPU обработка
        cudaEventRecord(gStop);
        cudaEventSynchronize(gStop);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, gStart, gStop);    // время GPU

        std::cout << "GPU time: " << gpuTime << " ms\n";   //Вывод времени

        // ---------- HYBRID ----------
        int mid = N / 2;                                  // граница массива
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float),
                               cudaMemcpyHostToDevice)); // обновляем данные

        auto hStart = std::chrono::high_resolution_clock::now();

        gpuProcessHalf<<<(N - mid + 255) / 256, 256>>>(d_arr, mid, N); // GPU половина
        cpuProcessPart(h_arr, 0, mid);                    // CPU половина

        CUDA_CHECK(cudaDeviceSynchronize());               // ждём GPU

        auto hEnd = std::chrono::high_resolution_clock::now();

        std::cout << "Hybrid time: "
                  << std::chrono::duration<double, std::milli>(hEnd - hStart).count()   //Вывод времени
                  << " ms\n";

        cudaFree(d_arr);
        delete[] h_arr;
    }

    return 0;
}
