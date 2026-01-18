#include <iostream>
#include <cuda_runtime.h>

#define N 1000000       // размер массива, который будем умножать

// Макрос для проверки ошибок CUDA
// Если CUDA вызов возвращает ошибку, выводим сообщение и прекращаем выполнение
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


// Ядро: глобальная память (каждый поток умножает один элемент)

__global__ void multiplyGlobal(float* arr, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисляем уникальный индекс для каждого потока
    if (idx < n)                                     // проверяем, чтобы не выйти за предел массива
        arr[idx] *= factor;                          // умножаем элемент на заданный коэффициент
}

int main() {

    // Создаем массив на CPU

    float *h_arr = new float[N]; // основной массив на CPU
    float *d_arr;                // указатель на массив на GPU

    // Заполняем массив единицами
    for (int i = 0; i < N; ++i)
        h_arr[i] = 1.0f;


    // Выделяем память на GPU

    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float)));

    // Копируем данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    // Создаем таймеры для измерения времени работы на GPU
    cudaEvent_t start, stop;
    float elapsed;  // переменная для хранения времени выполнения


    // ВАРИАНТ 1: неоптимальный блок
    // Маленький блок потоков (64) – плохо использует GPU

    int block_bad = 64;
    int grid_bad = (N + block_bad - 1) / block_bad; // вычисляем сколько блоков нужно

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));             // запускаем таймер

    multiplyGlobal<<<grid_bad, block_bad>>>(d_arr, 2.0f, N); // запуск ядра
    CUDA_CHECK(cudaGetLastError());               // проверяем ошибки после запуска

    CUDA_CHECK(cudaEventRecord(stop));            // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));       // ждем, пока GPU завершит работу
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); // вычисляем время

    // Копируем результат с GPU на CPU для проверки
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Выводим один элемент массива и время выполнения
    std::cout << "Bad config sample: " << h_arr[0]
              << " | Block size: " << block_bad
              << " | Time: " << elapsed << " ms" << std::endl;


    // ВАРИАНТ 2: оптимальный блок
    // Блок размером 256 потоков хорошо использует GPU

    // Сброс массива на 1.0
    for (int i = 0; i < N; ++i) h_arr[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_opt = 256;                              // оптимальный размер блока для Tesla T4
    int grid_opt = (N + block_opt - 1) / block_opt;   // вычисляем количество блоков

    CUDA_CHECK(cudaEventRecord(start));               // стартуем таймер

    multiplyGlobal<<<grid_opt, block_opt>>>(d_arr, 2.0f, N); // запуск ядра
    CUDA_CHECK(cudaGetLastError());                    // проверяем ошибки

    CUDA_CHECK(cudaEventRecord(stop));                // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));           // ждем завершения
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); // время выполнения

    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат на CPU

    // Выводим один элемент массива и время выполнения
    std::cout << "Optimized config sample: " << h_arr[0]
              << " | Block size: " << block_opt
              << " | Time: " << elapsed << " ms" << std::endl;


    // Освобождаем память на GPU и CPU

    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;

    return 0;
}
