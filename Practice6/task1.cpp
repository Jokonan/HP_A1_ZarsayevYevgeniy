#include <CL/cl.h>          // основной заголовок OpenCL
#include <iostream>         // ввод и вывод в консоль
#include <vector>           // удобные динамические массивы
#include <fstream>          // чтение файла kernel.cl
#include <chrono>           // измерение времени
#include <cstdlib>          // rand()
#include <cmath>            // fabs() для сравнения float


// Макрос для проверки ошибок OpenCL
#define CL_CHECK(err)                                   \
    if (err != CL_SUCCESS) {                            \
        std::cerr << "OpenCL error: " << err            \
                  << " at line " << __LINE__ << std::endl; \
        exit(1);                                        \
    }

int main() {

    const int N = 1 << 20;              // размер массивов (примерно 1 миллион элементов)
    const size_t bytes = N * sizeof(float); // размер массива в байтах

    // ===== CPU данные =====
    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N); // входные и выходные массивы

    // Заполняем входные массивы случайными числами
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;  // случайное число от 0 до 1
        B[i] = static_cast<float>(rand()) / RAND_MAX;  // случайное число от 0 до 1
    }

    // ===== CPU версия (последовательная) =====
    auto cpu_start = std::chrono::high_resolution_clock::now(); // старт замера времени CPU

    for (int i = 0; i < N; ++i) {
        C_cpu[i] = A[i] + B[i];          // обычное сложение на CPU
    }

    auto cpu_end = std::chrono::high_resolution_clock::now(); // конец замера времени CPU
    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // время в мс

    // ===== OpenCL инициализация =====
    cl_int err;                          // переменная для хранения кодов ошибок

    cl_platform_id platform;             // платформа OpenCL (например NVIDIA)
    cl_device_id device;                 // устройство OpenCL (GPU)

    // Получаем первую доступную платформу
    err = clGetPlatformIDs(1, &platform, nullptr);
    CL_CHECK(err);                       // проверяем, что ошибок нет

    // Получаем GPU устройство
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CL_CHECK(err);                       // проверяем, что GPU найден

    // Создаем контекст OpenCL
    cl_context context =
        clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);                       // проверяем успешность создания

    // Создаем очередь команд с возможностью замера времени
    cl_command_queue queue =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);                       // проверяем успешность создания очереди

    // ===== Загружаем kernel.cl =====
    std::ifstream file("task1_kernel.cl");     // открываем файл с kernel-кодом
    std::string kernelSource(            // читаем весь файл в строку
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    const char* sourceStr = kernelSource.c_str(); // преобразуем в C-строку

    // Создаем программу OpenCL из исходного кода
    cl_program program =
        clCreateProgramWithSource(context, 1, &sourceStr, nullptr, &err);
    CL_CHECK(err);                       // проверяем создание программы

    // Компилируем программу
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {             // если компиляция не удалась
        size_t logSize;
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize); // узнаем размер лога
        std::vector<char> log(logSize);  // выделяем память под лог
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG,
                              logSize, log.data(), nullptr); // получаем лог
        std::cerr << "Build log:\n" << log.data() << std::endl; // выводим ошибки
        exit(1);                          // завершаем программу
    }

    // Создаем kernel (ядро) по имени функции в kernel.cl
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CL_CHECK(err);                       // проверяем создание ядра

    // ===== Буферы OpenCL =====
    cl_mem d_A = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, A.data(), &err); // буфер для A
    CL_CHECK(err);

    cl_mem d_B = clCreateBuffer(context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, B.data(), &err); // буфер для B
    CL_CHECK(err);

    cl_mem d_C = clCreateBuffer(context,
                                CL_MEM_WRITE_ONLY,
                                bytes, nullptr, &err); // буфер для результата
    CL_CHECK(err);

    // Передаем аргументы в kernel
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A)); // аргумент A
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B)); // аргумент B
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C)); // аргумент C

    // ===== Запуск ядра =====
    size_t globalSize = N;               // общее количество потоков
    size_t localSize = 256;              // количество потоков в группе

    cl_event event;                      // событие для замера времени
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                 &globalSize, &localSize,
                                 0, nullptr, &event);
    CL_CHECK(err);                       // проверяем запуск kernel

    clWaitForEvents(1, &event);          // ждем завершения kernel

    // ===== Измерение времени GPU =====
    cl_ulong startTime, endTime;          // время начала и конца
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &startTime, nullptr);
    clGetEventProfilingInfo(event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &endTime, nullptr);

    double gpu_time =
        (endTime - startTime) * 1e-6;    // перевод из наносекунд в миллисекунды

    // ===== Копируем результат =====
    clEnqueueReadBuffer(queue, d_C, CL_TRUE,
                        0, bytes, C_gpu.data(),
                        0, nullptr, nullptr); // копируем результат с GPU

    // ===== Проверка корректности =====
    bool correct = true;                 // флаг корректности
    for (int i = 0; i < N; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-6) { // сравниваем CPU и GPU
            correct = false;
            break;
        }
    }

    // ===== Вывод =====
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;        // время CPU
    std::cout << "GPU OpenCL time: " << gpu_time << " ms" << std::endl; // время GPU
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
