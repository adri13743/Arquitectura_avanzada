#include <stdio.h>


#define N 4 // Número de filas de la matriz A
#define M 4 // Número de columnas de la matriz A (y número de elementos en el vector x)
#define threadsperblock 2

__global__ void matrixVectorProduct(float *A, float *x, float *y) {
    __shared__ float As[N]; // Memoria compartida para la matriz A
   
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(threadIdx.x == 0){ // solo entra 1 hilo para hacer el vector shared
	    for (int a = 0;a < N;a++){
	    	As[a]=x[a];
	    }
    }
    __syncthreads();
    printf("Bloque: %d, Hilo: %d \n ", blockIdx.x, threadIdx.x);
    if (col < N) {
        // Cargar la fila correspondiente de la matriz A en memoria compartida
        for (int row = 0; row < N; ++row) {
            printf("Bloque: %d, Hilo: %d, A[col * N + row]: %f, As[col]: %f  \n ", blockIdx.x, threadIdx.x, A[row * N + col], As[row]);
            sum += A[row * N + col] * As[row];
        }
        // Escribir el resultado de la suma en el vector y
        y[col] = sum;
    }
}




int main() {
    float *A, *x, *y; // Matriz A, vector x, y vector resultante
    float *d_A, *d_x, *d_y; // Punteros para la memoria en el dispositivo (GPU)
    float *d_A_transpose;
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
    float *A_transpose = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A_transpose[j * N + i] = A[i * M + j];
        }
    }
    cudaMalloc(&d_A_transpose, M * N * sizeof(float));


    // Copiar datos desde el host al dispositivo para la matriz transpuesta
    
    // Asignar memoria en el dispositivo (GPU)
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));


    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_A_transpose, A_transpose, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);


    // Definir las dimensiones del grid y los bloques
    dim3 gridSize(N/threadsperblock); // Un hilo por bloque
    // Definir las dimensiones del grid y los bloques
    // dim3 gridSize(N/threadsperblock); // Número de bloques
    //int threadsPerBlock = (N + 1) / 4; // Número de hilos por bloque



     printf("%d,%d,%d",gridSize.x, gridSize.y, gridSize.z);


    // Llamar al kernel
    matrixVectorProduct<<<gridSize, threadsperblock>>>(d_A_transpose, d_x, d_y);


    // Copiar resultado desde el dispositivo al host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);


    // Imprimir resultado
    printf("Vector resultado: ");
    printf("\n");
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




