#include <mpi.h>          // MPI
#include <iostream>       // вывод
#include <vector>         // массивы
#include <numeric>        // accumulate

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);                 // инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // количество процессов

    int N = 10000000;                       // увеличенный размер массива
    int localN = N / size;                  // часть на процесс

    std::vector<float> localArr(localN, 1.0f); // локальный массив заполнен 1.0

    MPI_Barrier(MPI_COMM_WORLD);             // синхронизация процессов
    double start = MPI_Wtime();              // старт таймера

    float localSum = std::accumulate(localArr.begin(), localArr.end(), 0.0f); // локальная сумма

    float globalSum = 0.0f;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // сбор суммы

    double end = MPI_Wtime();                // конец таймера

    if (rank == 0) {
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Sum: " << globalSum << std::endl;
        std::cout << "Time: " << (end - start) * 1000 << " ms\n"; // вывод в мс
    }

    MPI_Finalize();
    return 0;
}
