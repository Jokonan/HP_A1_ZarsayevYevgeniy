#include <iostream>
#include <cuda_runtime.h>

#define N 1000000        // размер массива
#define BLOCK_SIZE 256   // количество потоков в одном блоке


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


// Ядро 1: глобальная память
// Каждый поток обрабатывает один элемент массива напрямую в глобальной памяти

__global__ void multiplyGlobal(float* arr, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // уникальный индекс потока в массиве
    if (idx < n)                                     // проверяем, чтобы не выйти за пределы массива
        arr[idx] *= factor;                          // умножаем элемент на заданный коэффициент
}


// Ядро 2: shared память
// Используем быструю память блока для временного хранения элементов
// Это ускоряет доступ к данным по сравнению с глобальной памятью

__global__ void multiplyShared(float* arr, float factor, int n) {
    __shared__ float temp[BLOCK_SIZE];               // выделяем shared память на блок

    int tid = threadIdx.x;                           // индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + tid;        // глобальный индекс элемента

    if (idx < n)
        temp[tid] = arr[idx];                       // копируем элемент в shared память

    __syncthreads();                                // ждем, пока все потоки блока загрузят данные

    if (idx < n)
        temp[tid] *= factor;                        // выполняем умножение в быстрой памяти

    __syncthreads();                                // ждем, пока все потоки завершат умножение

    if (idx < n)
        arr[idx] = temp[tid];                       // записываем результат обратно в глобальную память
}

int main() {
    float *d_arr;                 // массив на GPU
    float *h_arr = new float[N];  // массив на CPU (основной массив)

    // ------------------------
    // Инициализация массива на CPU
    // ------------------------
    for (int i = 0; i < N; ++i)
        h_arr[i] = 1.0f;          // заполняем массив единицами

    // ------------------------
    // Выделяем память на GPU
    // ------------------------
    CUDA_CHECK(cudaMalloc(&d_arr, N * sizeof(float)));

    // Копируем массив с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------
    // Настройка блоков и сетки потоков
    // ------------------------
    dim3 block(BLOCK_SIZE);                        // количество потоков в блоке
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // количество блоков, чтобы покрыть все элементы массива

    cudaEvent_t start, stop;    // таймеры для измерения времени работы ядра
    float timeGlobal, timeShared;


    // Версия 1: глобальная память
   
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));           // стартуем таймер

    multiplyGlobal<<<grid, block>>>(d_arr, 2.0f, N); // запуск ядра на GPU
    CUDA_CHECK(cudaGetLastError());                  // проверка ошибок после запуска ядра

    CUDA_CHECK(cudaEventRecord(stop));              // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));         // ждем завершения работы GPU
    CUDA_CHECK(cudaEventElapsedTime(&timeGlobal, start, stop)); // вычисляем время выполнения

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Global memory result sample: " << h_arr[0] << std::endl;
    std::cout << "Time (global memory): " << timeGlobal << " ms" << std::endl;


    // Версия 2: shared память

    // Сброс массива на CPU на 1.0 перед запуском нового ядра
    for (int i = 0; i < N; ++i) h_arr[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));                        // старт таймера
    multiplyShared<<<grid, block>>>(d_arr, 2.0f, N);           // запуск ядра с shared памятью
    CUDA_CHECK(cudaGetLastError());                             // проверка ошибок

    CUDA_CHECK(cudaEventRecord(stop));                         // останавливаем таймер
    CUDA_CHECK(cudaEventSynchronize(stop));                    // ждем завершения работы GPU
    CUDA_CHECK(cudaEventElapsedTime(&timeShared, start, stop)); // вычисляем время выполнения

    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат обратно на CPU
    std::cout << "Shared memory result sample: " << h_arr[0] << std::endl;
    std::cout << "Time (shared memory): " << timeShared << " ms" << std::endl;


    // Освобождаем память на GPU и CPU

    CUDA_CHECK(cudaFree(d_arr));
    delete[] h_arr;

    return 0;
}
