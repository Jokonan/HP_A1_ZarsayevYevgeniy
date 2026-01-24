
// Ядро OpenCL для поэлементного сложения двух массивов
__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C)
{
    int id = get_global_id(0); // глобальный индекс текущего work-item
    C[id] = A[id] + B[id];     // складываем элементы массивов
}
