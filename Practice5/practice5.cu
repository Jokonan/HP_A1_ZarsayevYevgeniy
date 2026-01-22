#include <iostream>          // для ввода/вывода
#include <cuda_runtime.h>    // функции CUDA
#include <cstdlib>           // rand(), srand()
#include <ctime>             // генерация времени для таймера

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

// ======================================================
// CPU версия стека
// ======================================================
struct CpuStack {
    int* data;       // массив для элементов стека
    int top;         // индекс вершины
    int capacity;    // размер стека

    void init(int* buffer, int size) {
        data = buffer;  // указываем память
        top = -1;       // стек пуст
        capacity = size;
    }

    bool push(int value) {
        if (top + 1 < capacity) { // проверка переполнения
            data[++top] = value;  // добавляем элемент
            return true;
        }
        return false; // стек переполнен
    }

    bool pop(int* value) {
        if (top >= 0) {          // проверка пустого стека
            *value = data[top--]; // извлекаем элемент
            return true;
        }
        return false; // стек пуст
    }
};

// ======================================================
// CPU версия очереди
// ======================================================
struct CpuQueue {
    int* data;       // массив элементов
    int head;        // начало очереди
    int tail;        // конец очереди
    int capacity;    // размер очереди

    void init(int* buffer, int size) {
        data = buffer;  // память
        head = 0;       // начало
        tail = 0;       // конец
        capacity = size;
    }

    bool enqueue(int value) {
        if (tail < capacity) {  // проверка переполнения
            data[tail++] = value; // добавляем элемент
            return true;
        }
        return false; // очередь переполнена
    }

    bool dequeue(int* value) {
        if (head < tail) {       // проверка пустой очереди
            *value = data[head++]; // забираем элемент
            return true;
        }
        return false; // очередь пуста
    }
};

// ======================================================
// GPU стек
// ======================================================
struct Stack {
    int* data;      // массив для элементов
    int top;        // вершина стека
    int capacity;   // размер стека

    __device__ void init(int* buffer, int size) {
        data = buffer;  // память для элементов
        top = -1;       // стек пуст
        capacity = size;
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1); // атомарно увеличиваем top
        if (pos < capacity) {
            data[pos] = value;        // добавляем элемент
            return true;
        }
        return false; // стек переполнен
    }

    __device__ bool pop(int* value) {
        int pos = atomicSub(&top, 1); // атомарно уменьшаем top
        if (pos >= 0) {
            *value = data[pos];       // извлекаем элемент
            return true;
        }
        return false; // стек пуст
    }
};

// ======================================================
// GPU очередь
// ======================================================
struct Queue {
    int* data;       // массив элементов
    int head;        // начало
    int tail;        // конец
    int capacity;    // размер

    __device__ void init(int* buffer, int size) {
        data = buffer;  // память
        head = 0;       // начало
        tail = 0;       // конец
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); // атомарно получаем позицию
        if (pos < capacity) {
            data[pos] = value;         // добавляем элемент
            return true;
        }
        return false; // переполнение
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1); // атомарно получаем позицию
        if (pos < tail) {
            *value = data[pos];        // извлекаем элемент
            return true;
        }
        return false; // очередь пуста
    }
};

// ======================================================
// GPU MPMC очередь
// ======================================================
struct MpmcQueue {
    int* data;       // массив элементов
    int head;        // индекс головы
    int tail;        // индекс хвоста
    int capacity;    // размер очереди

    __device__ void init(int* buffer, int size) {
        data = buffer;  // память
        head = 0;       // начало
        tail = 0;       // конец
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); // атомарно получаем позицию
        if (pos < capacity) {
            data[pos] = value;         // добавляем
            return true;
        }
        return false; // переполнение
    }

    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1); // атомарно получаем позицию
        if (pos < tail) {
            *value = data[pos];        // извлекаем
            return true;
        }
        return false; // пусто
    }
};

// ======================================================
// GPU очередь с shared memory
// ======================================================
struct SharedQueue {
    int* data;        // массив элементов
    int head;         // начало
    int tail;         // конец
    int capacity;     // размер

    __device__ void init(int* buffer, int size) {
        data = buffer;  // память
        head = 0;       // начало
        tail = 0;       // конец
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); // атомарно получаем позицию
        if (pos < capacity) {
            data[pos] = value;         // добавляем
            return true;
        }
        return false; // переполнение
    }

    __device__ bool dequeue(int* value) {
        __shared__ int buffer[256];     // shared memory
        int tid = threadIdx.x;

        int pos = atomicAdd(&head, 1);  // атомарно получаем позицию
        if (pos < tail) {
            buffer[tid] = data[pos];    // временный буфер
            __syncthreads();            // синхронизация
            *value = buffer[tid];       // извлекаем
            return true;
        }
        return false; // пусто
    }
};

// ======================================================
// Ядра для стека и очередей
// ======================================================
__global__ void stackKernel(Stack* stack) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // индекс потока
    stack->push(tid);      // добавляем
    __syncthreads();       // ждём завершения push
    int value;
    stack->pop(&value);    // извлекаем
}

__global__ void queueKernel(Queue* queue) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    queue->enqueue(tid);    // добавляем
    __syncthreads();
    int value;
    queue->dequeue(&value); // извлекаем
}

__global__ void mpmcKernel(MpmcQueue* queue) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    queue->enqueue(tid);    // MPMC enqueue
    __syncthreads();
    int value;
    queue->dequeue(&value);  // MPMC dequeue
}

__global__ void sharedQueueKernel(SharedQueue* queue) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    queue->enqueue(tid);     // enqueue
    __syncthreads();
    int value;
    queue->dequeue(&value);  // dequeue
}

// ======================================================
// ОСНОВНАЯ ФУНКЦИЯ
// ======================================================
int main() {

    const int N = 1024 * 1024;         // количество элементов
    const int threads = 256;           // потоков на блок
    const int blocks = (N + threads - 1) / threads; // блоков

    int tmp; // для pop/dequeue

    // -----------------------
    // CPU стек
    // -----------------------
    int* cpuStackData = new int[N];
    CpuStack cpuStack;
    cpuStack.init(cpuStackData, N);

    clock_t startCPUStack = clock();
    for (int i = 0; i < N; ++i) cpuStack.push(i);
    for (int i = 0; i < N; ++i) cpuStack.pop(&tmp);
    clock_t endCPUStack = clock();
    double cpuStackTime = 1000.0 * (endCPUStack - startCPUStack) / CLOCKS_PER_SEC;
    std::cout << "CPU stack time: " << cpuStackTime << " ms" << std::endl;
    delete[] cpuStackData;

    // -----------------------
    // CPU очередь
    // -----------------------
    int* cpuQueueData = new int[N];
    CpuQueue cpuQueue;
    cpuQueue.init(cpuQueueData, N);

    clock_t startCPUQueue = clock();
    for (int i = 0; i < N; ++i) cpuQueue.enqueue(i);
    for (int i = 0; i < N; ++i) cpuQueue.dequeue(&tmp);
    clock_t endCPUQueue = clock();
    double cpuQueueTime = 1000.0 * (endCPUQueue - startCPUQueue) / CLOCKS_PER_SEC;
    std::cout << "CPU queue time: " << cpuQueueTime << " ms" << std::endl;
    delete[] cpuQueueData;

    // -----------------------
    // GPU память
    // -----------------------
    Stack* d_stack;
    Queue* d_queue;
    MpmcQueue* d_mpmc;
    SharedQueue* d_shared;
    int* d_buffer;

    CUDA_CHECK(cudaMalloc(&d_buffer, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stack, sizeof(Stack)));
    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(Queue)));
    CUDA_CHECK(cudaMalloc(&d_mpmc, sizeof(MpmcQueue)));
    CUDA_CHECK(cudaMalloc(&d_shared, sizeof(SharedQueue)));

    // Таймер CUDA
    cudaEvent_t start, stop;
    float timeMs;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -----------------------
    // GPU стек
    // -----------------------
    CUDA_CHECK(cudaEventRecord(start));
    stackKernel<<<blocks, threads>>>(d_stack);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));
    std::cout << "GPU stack time: " << timeMs << " ms" << std::endl;

    // -----------------------
    // GPU очередь
    // -----------------------
    CUDA_CHECK(cudaEventRecord(start));
    queueKernel<<<blocks, threads>>>(d_queue);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));
    std::cout << "GPU queue time: " << timeMs << " ms" << std::endl;

    // -----------------------
    // MPMC очередь
    // -----------------------
    CUDA_CHECK(cudaEventRecord(start));
    mpmcKernel<<<blocks, threads>>>(d_mpmc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));
    std::cout << "GPU MPMC queue time: " << timeMs << " ms" << std::endl;

    // -----------------------
    // Shared memory очередь
    // -----------------------
    CUDA_CHECK(cudaEventRecord(start));
    sharedQueueKernel<<<blocks, threads>>>(d_shared);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));
    std::cout << "GPU Shared queue time: " << timeMs << " ms" << std::endl;

    // -----------------------
    // Освобождение GPU памяти
    // -----------------------
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_stack));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_mpmc));
    CUDA_CHECK(cudaFree(d_shared));

    return 0;
}
