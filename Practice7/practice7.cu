#include <iostream>          // ввод и вывод в консоль
#include <vector>            // контейнер vector
#include <cstdlib>           // rand(), RAND_MAX
#include <cuda_runtime.h>    // CUDA API
#include <chrono>            // таймер времени

using namespace std;         // пространство имён std


   // ================= CPU РЕАЛИЗАЦИИ =================

// CPU редукция суммы
float cpuReduce(const vector<float>& data)
{
    float sum = 0.0f;                    // аккумулятор суммы
    for (float v : data)                 // проход по всем элементам
        sum += v;                        // добавление элемента
    return sum;                          // возврат результата
}

// CPU префиксная сумма
void cpuScan(const vector<float>& in, vector<float>& out)
{
    out[0] = 0.0f;                       // первый элемент равен 0
    for (size_t i = 1; i < in.size(); i++)
        out[i] = out[i - 1] + in[i - 1]; // накопление суммы
}


   //ЗАДАНИЕ 1: РЕДУКЦИЯ

// редукция суммы
__global__ void reduceSum(const float* input, float* output, int n)
{
    extern __shared__ float s[];                         // shared memory блока
    int tid = threadIdx.x;                             // индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid;               // глобальный индекс элемента
    s[tid] = (i < n) ? input[i] : 0.0f;                 // загрузка элемента или 0
    __syncthreads();                               // ожидание всех потоков

    for (int step = blockDim.x / 2; step > 0; step >>= 1) // поэтапная редукция
    {
        if (tid < step) s[tid] += s[tid + step];          // сложение элементов
        __syncthreads();                              // синхронизация шага
    }

    if (tid == 0) output[blockIdx.x] = s[0];    // запись суммы блока
}


   //ЗАДАНИЕ 2: ПРЕФИКСНАЯ СУММА

__global__ void blellochScan(float* data)
{
    extern __shared__ float temp[];                        // shared memory
    int tid = threadIdx.x;                             // индекс потока
    int offset = 1;                            // начальный шаг

    temp[2 * tid]     = data[2 * tid];             // загрузка первого элемента
    temp[2 * tid + 1] = data[2 * tid + 1];              // загрузка второго элемента

    // upsweep фаза
    for (int d = blockDim.x; d > 0; d >>= 1)            // движение к корню
    {
        __syncthreads();                    // синхронизация
        if (tid < d)
        {
            int i1 = offset * (2 * tid + 1) - 1;           // левый узел
            int i2 = offset * (2 * tid + 2) - 1;           // правый узел
            temp[i2] += temp[i1];                  // накопление суммы
        }
        offset <<= 1;                  // увеличение шага
    }

    if (tid == 0) temp[2 * blockDim.x - 1] = 0;        // exclusive scan

    // downsweep фаза
    for (int d = 1; d <= blockDim.x; d <<= 1)      // распространение сумм
    {
        offset >>= 1;                       // уменьшение шага
        __syncthreads();                    // синхронизация
        if (tid < d)
        {
            int i1 = offset * (2 * tid + 1) - 1;           // левый индекс
            int i2 = offset * (2 * tid + 2) - 1;         // правый индекс
            float t = temp[i1];                         // временное значение
            temp[i1] = temp[i2];                   // перестановка
            temp[i2] += t;                   // обновление суммы
        }
    }

    __syncthreads();                              // финальная синхронизация
    data[2 * tid]     = temp[2 * tid];               // запись первого результата
    data[2 * tid + 1] = temp[2 * tid + 1];            // запись второго результата
}


   //ЗАДАНИЕ 3: ПРОИЗВОДИТЕЛЬНОСТЬ

int main()
{
    vector<int> sizes = {1024, 1000000, 10000000};         // размеры массивов
    vector<int> blocks = {128, 256, 512};                  // размеры блоков

    for (int N : sizes)                      // перебор размеров
    {
        cout << "\nArray size: " << N << endl;        // вывод размера

        vector<float> h(N);
        for (int i = 0; i < N; i++)                 // заполнение массива
            h[i] = rand() / (float)RAND_MAX;           // случайные значения

        /* ---------- CPU BENCHMARK ---------- */

        auto c1 = chrono::high_resolution_clock::now();   // старт CPU редукции
        float cpu_sum = cpuReduce(h);                     // CPU редукция
        auto c2 = chrono::high_resolution_clock::now();   // конец

        double cpu_reduce_time =
            chrono::duration<double, milli>(c2 - c1).count();

        cout << "[CPU REDUCTION] Time (ms): "
             << cpu_reduce_time << endl;                  // вывод CPU времени

        if (N >= 2)
        {
            vector<float> cpu_scan_out(N);                // выход scan

            auto c3 = chrono::high_resolution_clock::now(); // старт CPU scan
            cpuScan(h, cpu_scan_out);                       // CPU scan
            auto c4 = chrono::high_resolution_clock::now(); // конец

            double cpu_scan_time =
                chrono::duration<double, milli>(c4 - c3).count();

            cout << "[CPU SCAN]      Time (ms): "
                 << cpu_scan_time << endl;
        }

        float* d_in;                        // указатель GPU
        cudaMalloc(&d_in, N * sizeof(float));              // выделение памяти
        cudaMemcpy(d_in, h.data(), N * sizeof(float),
                   cudaMemcpyHostToDevice);                // копирование на GPU

        for (int block : blocks)              // перебор размеров блоков
        {
            /* ---------- РЕДУКЦИЯ ---------- */
            int grid = (N + block - 1) / block;         // число блоков
            float* d_out;                        // выход редукции
            cudaMalloc(&d_out, grid * sizeof(float));      // память под результат

            auto t1 = chrono::high_resolution_clock::now();// старт таймера
            reduceSum<<<grid, block, block * sizeof(float)>>>(d_in, d_out, N); // запуск ядра
            cudaDeviceSynchronize();                        // ожидание GPU
            auto t2 = chrono::high_resolution_clock::now();// конец таймера

            double t_reduce =
                chrono::duration<double, milli>(t2 - t1).count(); // время редукции
            cout << "[REDUCTION] Block size " << block
                 << " -> GPU time (ms): " << t_reduce << endl;    // вывод времени

            cudaFree(d_out);            // освобождение памяти

            /* ---------- PREFIX SUM ---------- */
            if (N >= 2 * block)                       // проверка размера
            {
                auto t3 = chrono::high_resolution_clock::now(); // старт таймера
                blellochScan<<<1, block,
                    2 * block * sizeof(float)>>>(d_in);    // запуск scan
                cudaDeviceSynchronize();              // ожидание GPU
                auto t4 = chrono::high_resolution_clock::now(); // конец таймера

                double t_scan =
                    chrono::duration<double, milli>(t4 - t3).count(); // время scan
                cout << "[SCAN]      Block size " << block
                     << " -> GPU time (ms): " << t_scan << endl;    // вывод времени
            }
        }

        cudaFree(d_in);         // освобождение памяти GPU
    }

    return 0;
}
