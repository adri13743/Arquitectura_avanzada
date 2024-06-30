#include <stdio.h>


#define N 4 // Número de filas de la matriz A
#define M 4 // Número de columnas de la matriz A (y número de elementos en el vector x)
#define threadsperblock 2

__global__ void matrixVectorProduct(float *A, float *x, float *y, float *partialSum) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Índice del hilo en el grid
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int inicio =  (N/threadsperblock);
    //int fin = (M/threadsperblock);
    float sum = 0.0;
    if (row < N && col < M) {
        int columna = col*threadsperblock;
        int fila = row;
        //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d, blockIdx.y: %d, blockDim.y: %d, threadIdx.x: %d, \n", blockIdx.x, threadIdx.x, row,col,fila,columna,blockIdx.y,blockDim.y,threadIdx.x);
        for (int j = 0; j < blockDim.x; j++) {
            sum += A[fila * M + ((columna)+j)] * x[columna+j];
            //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d ,fila * M + (columna+j): %f, columna+j: %f\n", blockIdx.x, threadIdx.x, row,col,fila,columna,A[fila * M + ((columna*threadsperblock)+j)],x[columna+j]);
            if(j == blockDim.x-1){
                __syncthreads();
                //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d ,fila * M + (columna+j): %d, columna+j: %d, sum: %f\n", blockIdx.x, threadIdx.x, row,col,fila,columna,fila * M + (columna+j),columna+j,sum);
            }
        }
        
        atomicAdd(&y[row], sum);
    }
}


int main() {
    float *A, *x, *y; // Matriz A, vector x, y vector resultante
    float *d_A, *d_x, *d_y; // Punteros para la memoria en el dispositivo (GPU)
    float *partialSum;
    // Asignar memoria en el host (CPU)
    A = (float*)malloc(N * M * sizeof(float));
    partialSum = (float*)malloc(N * M * sizeof(float));
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
    cudaMalloc(&partialSum, N * M * sizeof(float));
    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));


    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(partialSum, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);


    // Definir las dimensiones del grid y los bloques
    dim3 gridSize(N/threadsperblock,M/threadsperblock);



    printf("%d,%d,%d \n",gridSize.x, gridSize.y, gridSize.z);


    // Llamar al kernel
    matrixVectorProduct<<<gridSize, threadsperblock>>>(d_A, d_x, d_y,partialSum);




    // Copiar resultado desde el dispositivo al host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);


    // Imprimir resultado
    printf("Vector resultado: ");
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



