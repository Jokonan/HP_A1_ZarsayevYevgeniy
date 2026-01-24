

// Ядро OpenCL для умножения матриц
__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     const int N,
                     const int M,
                     const int K) {

    int row = get_global_id(0); // индекс строки в матрице C
    int col = get_global_id(1); // индекс столбца в матрице C

    if (row < N && col < K) {   // проверяем, что не выходим за границы
        float sum = 0.0f;       // локальная переменная для суммы
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col]; // стандартная формула умножения
        }
        C[row * K + col] = sum; // записываем результат в матрицу C
    }
}
