#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ================= UTILS =================
device host inline int imin(int a, int b) { return (a < b) ? a : b; }

std::vector<int> generateData(int n) {
    std::vector<int> v(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1000000);
    for (int& x : v) x = dis(gen);
    return v;
}

// ================= GPU: сортировка блоков (bubble sort для учебной демонстрации) =================
global void gpuBlockSort(int* data, int n, int chunkSize) {
    int blockStart = blockIdx.x * chunkSize;
    int blockEnd = imin(blockStart + chunkSize, n);
    for (int i = blockStart; i < blockEnd; i++) {
        for (int j = i + 1; j < blockEnd; j++) {
            if (data[i] > data[j]) {
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
            }
        }
    }
}

// ================= GPU: слияние пар блоков =================
global void gpuMergePairs(int* data, int n, int step) {
    int tid = blockIdx.x;
    int start = tid * step * 2;
    int mid = imin(start + step, n);
    int end = imin(start + step * 2, n);

    int i = start, j = mid, k = 0;
    extern shared int tmp[];
    while (i < mid && j < end) tmp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
    while (i < mid) tmp[k++] = data[i++];
    while (j < end) tmp[k++] = data[j++];
    for (int x = 0; x < k; x++) data[start + x] = tmp[x];
}

// ================= MAIN =================
int main() {
    std::vector<int> sizes = {10000, 100000};

    for (int n : sizes) {
        std::cout << "\n=== Array size: " << n << " ===\n";

        auto base = generateData(n);

        // -------- GPU --------
        int* d_data;
        cudaMalloc(&d_data, n * sizeof(int));
        cudaMemcpy(d_data, base.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int chunkSize = 1024;
        int blocks = (n + chunkSize - 1) / chunkSize;

        // ---------- Сортировка блоков ----------
        cudaEventRecord(start);
        gpuBlockSort<<<blocks, 1>>>(d_data, n, chunkSize);
        cudaDeviceSynchronize();

        // ---------- Параллельное слияние блоков ----------
        int mergeStep = chunkSize;
        while (mergeStep < n) {
            int mergeBlocks = (n + mergeStep * 2 - 1) / (mergeStep * 2);
            gpuMergePairs<<<mergeBlocks, 1, mergeStep * 2 * sizeof(int)>>>(d_data, n, mergeStep);
            cudaDeviceSynchronize();
            mergeStep *= 2;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);
        std::cout << "GPU Merge sort: " << gpuTime << " ms\n";

        // Проверка первых 10 элементов
        std::vector<int> sorted(n);
        cudaMemcpy(sorted.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "First 10 elements: ";
        for (int i = 0; i < 10; i++) std::cout << sorted[i] << " ";
        std::cout << "\n";

        cudaFree(d_data);
    }

    return 0;
}