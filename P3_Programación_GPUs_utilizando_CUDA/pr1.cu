#include <stdio.h>


#define N  4// Número de filas de la matriz A
#define M 4 // Número de columnas de la matriz A (y número de elementos en el vector x)
#define threadsperblock 2

__global__ void matrixVectorProduct(float *A, float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Índice del hilo en el grid
    printf("Bloque: %d, Hilo: %d , idx: %d \n ", blockIdx.x, threadIdx.x, idx);
    if (idx < N) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            sum += A[idx * M + j] * x[j];
        }
        y[idx] = sum;
    }
}


int main() {
    float *A, *x, *y; // Matriz A, vector x, y vector resultante
    float *d_A, *d_x, *d_y; // Punteros para la memoria en el dispositivo (GPU)


    // Asignar memoria en el host (CPU)
    A = (float*)malloc(N * M * sizeof(float));
    x = (float*)malloc(M * sizeof(float));
    y = (float*)malloc(N * sizeof(float));


    // Inicializar datos de la matriz A y el vector x
    for (int i = 0; i < N * M; i++) {
        A[i] = i;
    }
    for (int i = 0; i < M; i++) {
        x[i] = i;
    }


    // Asignar memoria en el dispositivo (GPU)
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));


    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);


    // Definir las dimensiones del grid y los bloques
    dim3 gridSize(N/threadsperblock); // Número de bloques
//    int threadsPerBlock = (N + 1) / 4; // Número de hilos por bloque



     printf("%d,%d,%d",gridSize.x, gridSize.y, gridSize.z);


    // Llamar al kernel
    matrixVectorProduct<<<gridSize, threadsperblock>>>(d_A, d_x, d_y);


    // Copiar resultado desde el dispositivo al host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n");
    printf("Matriz A: \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("Vector x: ");
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
    printf("Vector resultado: ");
    printf("\n");
    // Imprimir resultado
    for (int i = 0; i < N; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");


    // Liberar memoria en el dispositivo
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);


    // Liberar memoria en el host
    free(A);
    free(x);
    free(y);


    return 0;
}


