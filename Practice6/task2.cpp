#include <CL/cl.h>          // основной заголовок OpenCL
#include <iostream>         // ввод-вывод в консоль
#include <vector>           // динамические массивы (std::vector)
#include <fstream>          // чтение файлов (ядра OpenCL)
#include <chrono>           // измерение времени
#include <cstdlib>          // rand() для случайных чисел
#include <cmath>            // fabs() для проверки результатов

// Макрос для проверки ошибок OpenCL
#define CL_CHECK(err)                                   \
    if (err != CL_SUCCESS) {                            \
        std::cerr << "OpenCL error: " << err            \
                  << " at line " << __LINE__ << std::endl; \
        exit(1);                                        \
    }

int main() {

    // ===== Размеры матриц =====
    const int N = 512; // количество строк матрицы A и C
    const int M = 256; // количество столбцов матрицы A / строк матрицы B
    const int K = 512; // количество столбцов матрицы B и C

    const size_t sizeA = N * M * sizeof(float); // размер в байтах матрицы A
    const size_t sizeB = M * K * sizeof(float); // размер в байтах матрицы B
    const size_t sizeC = N * K * sizeof(float); // размер в байтах матрицы C

    // ===== CPU данные =====
    std::vector<float> A(N * M), B(M * K), C_cpu(N * K), C_gpu(N * K); // матрицы для CPU и GPU

    // Заполняем матрицы случайными числами от 0 до 1
    for (int i = 0; i < N*M; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX; // A
    for (int i = 0; i < M*K; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX; // B

    // ===== CPU последовательное умножение =====
    auto cpu_start = std::chrono::high_resolution_clock::now(); // старт таймера CPU
    for (int r = 0; r < N; ++r) {                    // строки матрицы A
        for (int c = 0; c < K; ++c) {                // столбцы матрицы B
            float sum = 0.0f;                         // временная переменная для суммы
            for (int i = 0; i < M; ++i)               // суммируем произведения элементов
                sum += A[r*M + i] * B[i*K + c];       // индексируем как одномерный массив
            C_cpu[r*K + c] = sum;                     // записываем результат в C_cpu
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now(); // конец таймера
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // время CPU в мс

    // ===== OpenCL инициализация =====
    cl_int err;                       // переменная для ошибок OpenCL
    cl_platform_id platform;           // платформа OpenCL
    cl_device_id device;               // устройство OpenCL (GPU)

    err = clGetPlatformIDs(1, &platform, nullptr); CL_CHECK(err);                // получаем платформу
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr); CL_CHECK(err); // получаем GPU

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); CL_CHECK(err); // создаем контекст
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); CL_CHECK(err); // создаем очередь команд с профилированием

    // ===== Загружаем kernel =====
    std::ifstream file("task2_kernel.cl"); // открываем файл ядра
    std::string kernelSource((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()); // читаем всё
    const char* sourceStr = kernelSource.c_str(); // получаем C-style строку для OpenCL

    cl_program program = clCreateProgramWithSource(context, 1, &sourceStr, nullptr, &err); CL_CHECK(err); // создаем программу из источника
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // компилируем программу
    if (err != CL_SUCCESS) { // если ошибка компиляции
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize); // размер лога
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr); // получаем лог
        std::cerr << "Build log:\n" << log.data() << std::endl; // выводим лог
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err); CL_CHECK(err); // создаем ядро matmul

    // ===== Буферы =====
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, A.data(), &err); CL_CHECK(err); // буфер для A
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, B.data(), &err); CL_CHECK(err); // буфер для B
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, nullptr, &err); CL_CHECK(err); // буфер для C

    // ===== Аргументы ядра =====
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A)); // передаем A
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B)); // передаем B
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C)); // передаем C
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));     // передаем N
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &M));     // передаем M
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &K));     // передаем K

    // ===== Запуск ядра =====
    size_t globalSize[2] = {static_cast<size_t>(N), static_cast<size_t>(K)}; // глобальная область работы (2D)
    size_t localSize[2]  = {16, 16}; // локальная рабочая группа 16x16

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, &event); CL_CHECK(err); // запуск ядра
    clWaitForEvents(1, &event); // ждем завершения

    // ===== Время GPU =====
    cl_ulong startTime, endTime;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, nullptr); // начало
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, nullptr);     // конец
    double gpu_time = (endTime - startTime) * 1e-6; // наносекунды → миллисекунды

    // ===== Копируем результат =====
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeC, C_gpu.data(), 0, nullptr, nullptr); // читаем данные с GPU

    // ===== Проверка корректности =====
    bool correct = true;
    for (int i = 0; i < N*K; ++i) { // сравниваем все элементы
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-4f) { // допускаем небольшую погрешность
            correct = false;
            break;
        }
    }

    // ===== Вывод =====
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU OpenCL time: " << gpu_time << " ms" << std::endl;
    std::cout << "Result correct: " << (correct ? "YES" : "NO") << std::endl;

    // ===== Очистка =====
   clReleaseMemObject(d_A);              // освобождаем буфер A
    clReleaseMemObject(d_B);              // освобождаем буфер B
    clReleaseMemObject(d_C);              // освобождаем буфер C
    clReleaseKernel(kernel);              // освобождаем kernel
    clReleaseProgram(program);            // освобождаем программу
    clReleaseCommandQueue(queue);         // освобождаем очередь
    clReleaseContext(context);            // освобождаем контекст

    return 0;
}
