#include <iostream>
#include <cuda_runtime.h>

#define N 1000000        // размер массива
#define BLOCK_SIZE 256   // стандартный размер блока потоков


// Макрос для проверки ошибок CUDA
// Если любой CUDA-вызов вернет ошибку, выводим сообщение и завершаем программу

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


// Ядро 1: коалесцированный доступ
// Каждый поток обрабатывает последовательные элементы массива
// Это пример оптимального доступа к глобальной памяти

__global__ void coalescedAccess(float* arr, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    if (idx < n)                                     // проверка границ массива
        arr[idx] *= factor;                           // умножаем элемент массива
}


// Ядро 2: некоалесцированный доступ
// Каждый поток обрабатывает элементы через "скачки"
// Чтобы показать эффект неэффективного доступа к глобальной памяти

__global__ void nonCoalescedAccess(float* arr, float factor, int n) {
    int tid = threadIdx.x;               // индекс потока внутри блока
    int block_size = blockDim.x;         // размер блока потоков
    int idx = blockIdx.x * block_size + tid; // начальный индекс элемента для потока

    // "Разбросанный" доступ: поток обрабатывает элементы через шаг = block_size * 4
    for (int i = idx; i < n; i += block_size * 4) {
        arr[i] *= factor;                // умножаем элемент массива
    }
}

int main() {
    float *h_arr = new float[N];  // массив на CPU
    float *d_arr;                 // массив на GPU


    // Инициализация массива единицами

    for (int i = 0; i < N; ++i)
        h_arr[i] = 1.0f;


    // Выделяем память на GPU и копируем данные

    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float))); // выделяем память на GPU
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice)); // копируем данные


    // Настройка сетки и блоков потоков

    dim3 block(BLOCK_SIZE);                       // блок из 256 потоков
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // количество блоков для покрытия всего массива

    cudaEvent_t start, stop;  // таймеры для измерения времени выполнения
    float elapsed;            // переменная для хранения времени


    // Коалесцированный доступ

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));  // старт таймера

    coalescedAccess<<<grid, block>>>(d_arr, 2.0f, N);  // запускаем ядро
    CUDA_CHECK(cudaGetLastError());                     // проверяем ошибки после запуска

    CUDA_CHECK(cudaEventRecord(stop));                  // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));             // ждем завершения всех потоков
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); // вычисляем время

    // Копируем результат обратно на CPU для проверки
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Coalesced access sample: " << h_arr[0]
              << " | Time: " << elapsed << " ms" << std::endl;


    // Некоалесцированный доступ

    // Сбрасываем массив на 1.0 перед запуском второго ядра
    for (int i = 0; i < N; ++i) h_arr[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));  // старт таймера

    nonCoalescedAccess<<<grid, block>>>(d_arr, 2.0f, N);  // запускаем ядро с "скачками"
    CUDA_CHECK(cudaGetLastError());                         // проверяем ошибки

    CUDA_CHECK(cudaEventRecord(stop));                      // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));                 // ждем завершения
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));// вычисляем время

    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Non-coalesced access sample: " << h_arr[0]
              << " | Time: " << elapsed << " ms" << std::endl;


    // Освобождаем память на GPU и CPU

    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;

    return 0;
}
