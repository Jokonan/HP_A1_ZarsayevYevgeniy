#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ======================================================
// Универсальная функция min для host/device
device host inline int imin(int a, int b) {
    return (a < b) ? a : b;
}

// ================= CPU SORTS =================

void cpuMerge(std::vector<int>& a, int l, int m, int r) {
    std::vector<int> tmp(r - l + 1);
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r) tmp[k++] = (a[i] < a[j]) ? a[i++] : a[j++];
    while (i <= m) tmp[k++] = a[i++];
    while (j <= r) tmp[k++] = a[j++];
    for (int x = 0; x < k; x++) a[l + x] = tmp[x];
}

void cpuMergeSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    cpuMergeSort(a, l, m);
    cpuMergeSort(a, m + 1, r);
    cpuMerge(a, l, m, r);
}

void cpuQuickSort(std::vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot = a[(l + r) / 2];
    int i = l, j = r;
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) std::swap(a[i++], a[j--]);
    }
    cpuQuickSort(a, l, j);
    cpuQuickSort(a, i, r);
}

void cpuHeapSort(std::vector<int>& a) {
    std::make_heap(a.begin(), a.end());
    std::sort_heap(a.begin(), a.end());
}

// ================= GPU MERGE SORT =================

// Локальная сортировка каждого блока 
global void gpuBlockMergeSort(int* data, int n, int chunkSize) {
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

// Слияние блоков по парам 
global void gpuMergePairs(int* data, int n, int step) {
    int tid = blockIdx.x;
    int start = tid * step * 2;
    int mid = imin(start + step, n);
    int end = imin(start + step * 2, n);

    int i = start, j = mid, k = 0;
    extern shared int tmp[];
    while (i < mid && j < end)
        tmp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
    while (i < mid) tmp[k++] = data[i++];
    while (j < end) tmp[k++] = data[j++];
    for (int x = 0; x < k; x++) data[start + x] = tmp[x];
}

// ================= GPU QUICK SORT =================

device void deviceQuickSort(int* data, int left, int right) {
    int i = left, j = right;
    int pivot = data[(left + right) / 2];
    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            int tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
            i++; j--;
        }
    }
    if (left < j) deviceQuickSort(data, left, j);
    if (i < right) deviceQuickSort(data, i, right);
}

global void gpuQuickSort(int* data, int chunkSize, int n) {
    int blockId = blockIdx.x;
    int start = blockId * chunkSize;
    int end = imin(start + chunkSize - 1, n - 1);
    if (start < n) deviceQuickSort(data, start, end);
}

// ================= GPU HEAP SORT =================

global void gpuHeapSort(int* data, int chunkSize, int n) {
    int blockId = blockIdx.x;
    int start = blockId * chunkSize;
    int end = imin(start + chunkSize, n);
    if (start < n) {
        for (int i = start; i < end; i++) {
            for (int j = i + 1; j < end; j++) {
                if (data[i] > data[j]) {
                    int t = data[i];
                    data[i] = data[j];
                    data[j] = t;
                }
            }
        }
    }
}

// ================= UTILS =================

std::vector<int> generateData(int n) {
    std::vector<int> v(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 1000000);
    for (int& x : v) x = dis(gen);
    return v;
}

// ================= MAIN =================

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};

    for (int n : sizes) {
        std::cout << "\n=== Array size: " << n << " ===\n";
        auto base = generateData(n);

        // -------- CPU SORTS --------
        {
            auto v = base;
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuMergeSort(v, 0, v.size() - 1);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "CPU Merge sort: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";
        }
        {
            auto v = base;
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuQuickSort(v, 0, v.size() - 1);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "CPU Quick sort: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";
        }
        {
            auto v = base;
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuHeapSort(v);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "CPU Heap sort: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";
        }

        // -------- GPU SORTS --------
        int* d_data;
        cudaMalloc(&d_data, n * sizeof(int));
        cudaMemcpy(d_data, base.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int chunkSize = 1024;
        int blocks = (n + chunkSize - 1) / chunkSize;

        // -------- GPU Quick Sort --------
        cudaEventRecord(start);
        gpuQuickSort<<<blocks, 1>>>(d_data, chunkSize, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);
        std::cout << "GPU Quick sort: " << gpuTime << " ms\n";

        // -------- GPU Heap Sort --------
        cudaEventRecord(start);
        gpuHeapSort<<<blocks, 1>>>(d_data, chunkSize, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        std::cout << "GPU Heap sort: " << gpuTime << " ms\n";

        // -------- GPU Merge Sort (блоки + merge пары) --------
        int mergeStep = chunkSize;
        cudaEventRecord(start);
        // локальные блоки
        gpuBlockMergeSort<<<blocks, 1>>>(d_data, n, chunkSize);
        cudaDeviceSynchronize();

        // последовательное слияние пар блоков
        while (mergeStep < n) {
            int mergeBlocks = (n + mergeStep * 2 - 1) / (mergeStep * 2);
            gpuMergePairs<<<mergeBlocks, 1, mergeStep * 2 * sizeof(int)>>>(d_data, n, mergeStep);
            cudaDeviceSynchronize();
            mergeStep *= 2;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        std::cout << "GPU Merge sort: " << gpuTime << " ms\n";

        cudaFree(d_data);
    }

    return 0;

}
