# PCA-EXP-5-Implement-Matrix-Multiplication-using-CUDA-C.-Find-the-elapsed-time.
# NAME: Yogavarma B
# REG NO: 2305002029

## Aim:
To implement Matrix Multiplication using CUDA C on the GPU and measure the elapsed execution time.

## Procedure:
1.Initialize two input matrices A and B with sample values.

2.Compute matrix multiplication on the CPU for verification.

3.Allocate memory for matrices on the GPU using cudaMalloc().

4.Copy input matrices from host memory to device memory using cudaMemcpy().

5.Launch the CUDA kernel to perform matrix multiplication in parallel.

6.Copy the result matrix from device memory back to host memory.

7.Display the result matrix and measure the elapsed execution time.

## PROGRAM
```
%%writefile matmul.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 4
#define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the matrix bounds
    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; ++k)
        {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

// Helper function for matrix multiplication on CPU (for verification)
void matrixMultiplyCPU(int a[][SIZE], int b[][SIZE], int c[][SIZE], int size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main()
{
    int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE] = {0};
    int cpu_c[SIZE][SIZE] = {0};  // Results from CPU calculation for verification
    int *dev_a, *dev_b, *dev_c;
    int size = SIZE * SIZE * sizeof(int);

    // Initialize matrices 'a' and 'b'
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Print input matrices
    printf("Matrix A:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", b[i][j]);
        }
        printf("\n");
    }

    // Compute on CPU for verification
    printf("Computing matrix multiplication on CPU for verification...\n");
    matrixMultiplyCPU(a, b, cpu_c, SIZE);

    printf("CPU Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", cpu_c[i][j]);
        }
        printf("\n");
    }

    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result: FAILED\n");
        return 1;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
        printf("Using CPU result as final\n");
        return 0;
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    // Allocate memory on the device
    cudaError_t err;

    err = cudaMalloc((void**)&dev_a, size);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&dev_b, size);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a);
        return 1;
    }

    err = cudaMalloc((void**)&dev_c, size);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a);
        cudaFree(dev_b);
        return 1;
    }

    // Copy input matrices from host to device memory
    err = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
        return 1;
    }

    err = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
        return 1;
    }

    // Initialize dev_c with zeros
    err = cudaMemset(dev_c, 0, size);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMemset: %s\n", cudaGetErrorString(err));
        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
        return 1;
    }

    // Set grid and block sizes
    dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Launch kernel
    printf("Launching CUDA kernel...\n");
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in kernel launch: %s\n", cudaGetErrorString(err));
        printf("Using CPU result as final\n");

        // Copy CPU result to c
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                c[i][j] = cpu_c[i][j];
            }
        }

        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

        // Print the CPU result as the final result
        printf("Final Result Matrix (from CPU):\n");
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                printf("%d ", c[i][j]);
            }
            printf("\n");
        }

        gettimeofday(&end, NULL);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

        return 0;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error in synchronize: %s\n", cudaGetErrorString(err));
        printf("Using CPU result as final\n");

        // Copy CPU result to c
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                c[i][j] = cpu_c[i][j];
            }
        }

        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

        // Print the CPU result as the final result
        printf("Final Result Matrix (from CPU):\n");
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                printf("%d ", c[i][j]);
            }
            printf("\n");
        }

        gettimeofday(&end, NULL);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

        return 0;
    }

    // Copy result matrix from device to host memory
    err = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
        printf("Using CPU result as final\n");

        // Copy CPU result to c
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                c[i][j] = cpu_c[i][j];
            }
        }

        cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

        // Print the CPU result as the final result
        printf("Final Result Matrix (from CPU):\n");
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                printf("%d ", c[i][j]);
            }
            printf("\n");
        }

        gettimeofday(&end, NULL);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

        return 0;
    }

    // Stop timer
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Print the result matrix
    printf("GPU Result Matrix:\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Print the elapsed time
    printf("Elapsed Time (GPU): %.6f seconds\n", elapsed_time);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
```
## Output:
<img width="1472" height="663" alt="{8D6BC894-970B-4FB5-A72C-2CAEE9CF9AB7}" src="https://github.com/user-attachments/assets/65624f1f-5caf-403b-9730-856c628cd60b" />

<img width="1237" height="762" alt="{CD6BDDC9-BC4E-4621-A353-710B087A78D9}" src="https://github.com/user-attachments/assets/1409be9e-4979-4f53-bbd4-5143de366a88" />

## Result:
Thus, Matrix Multiplication was successfully implemented using CUDA C. The matrix multiplication was performed on the GPU, and the elapsed execution time was measured. The GPU result matched the CPU result, verifying the correctness of the implementation.
