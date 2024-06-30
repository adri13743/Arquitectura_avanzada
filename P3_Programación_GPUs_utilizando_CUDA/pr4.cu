#include <stdio.h>


#define N 9 // Número de filas de la matriz A
#define M 9 // Número de columnas de la matriz A (y número de elementos en el vector x)
#define threadsperblock 3

__global__ void matrixVectorProduct(float *A_transpose, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // fila
    int col = blockIdx.y * blockDim.y + threadIdx.y; // columna
    int inicio =  (N/threadsperblock);
    __shared__ float As[N]; // Memoria compartida para la matriz A
    //int fin = (M/threadsperblock);
    float sum = 0.0;
    if(threadIdx.x == 0 && threadIdx.y== 0){ // solo entra 1 hilo para hacer el vector shared           
	    for (int a = 0;a < N;a++){
	    	As[a]=x[a];
	    }
    }
    __syncthreads();
    if (row < N && col < M) {
        int columna = row;
        int fila = col*threadsperblock;
        //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d, blockIdx.y: %d, blockDim.y: %d, threadIdx.x: %d, \n", blockIdx.x, threadIdx.x, row,col,fila,columna,blockIdx.y,blockDim.y,threadIdx.x);
        for (int j = 0; j < blockDim.x; j++) {
            sum += A_transpose[(fila+j) * M + columna] * As[fila+j];
            //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d , A_transpose[(fila+j) * M + columna]: %f,fila+j: %f\n", blockIdx.x, threadIdx.x, row,col,fila,columna,  A_transpose[(fila+j) * M + columna],x[fila+j]);
            if(j == blockDim.x-1){
                __syncthreads();
                //printf("Bloque: %d, Hilo: %d , row: %d, col: %d,fila: %d ,columna: %d ,fila * M + (columna+j): %d, columna+j: %d, sum: %f\n", blockIdx.x, threadIdx.x, row,col,fila,columna,fila * M + (columna+j),columna+j,sum);
            }
        }
        
        atomicAdd(&y[row], sum);
    }
}


int main() {
    float *A, *x, *y, *y_host; // Matriz A, vector x, y vector resultante
    float *d_A, *d_x, *d_y; // Punteros para la memoria en el dispositivo (GPU)
    float *d_A_transpose;
    // Asignar memoria en el host (CPU)
    A = (float*)malloc(N * M * sizeof(float));
    x = (float*)malloc(M * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    y_host = (float*)malloc(N * sizeof(float));

    // Inicializar datos de la matriz A y el vector x
    for (int i = 0; i < N * M; i++) {
        A[i] = i;
    }
    for (int i = 0; i < M; i++) {
        x[i] = i;
    }
    // Inicializar el vector resultado en el host
    for (int i = 0; i < N; i++) {
        y_host[i] = 0.0f;
    }
    float *A_transpose = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A_transpose[ i * M + j] = A[j * N + i];
            //printf("A_transpose= %f\n", A_transpose[i * M + j]);
        }
    }
    cudaMalloc(&d_A_transpose, M * N * sizeof(float));


    // Copiar datos desde el host al dispositivo para la matriz transpuesta
    
    // Asignar memoria en el dispositivo (GPU)

    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));


    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_A_transpose, A_transpose, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice);


    // Definir las dimensiones del grid y los bloques
    dim3 gridSize(M/threadsperblock,N/threadsperblock);



    printf("%d,%d,%d \n",gridSize.x, gridSize.y, gridSize.z);


    // Llamar al kernel
    matrixVectorProduct<<<gridSize, threadsperblock>>>(d_A_transpose, d_x, d_y);

    // Copiar resultado desde el dispositivo al host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        y_host[i] = 0.0f;
        for (int j = 0; j < M; j++) {
            y_host[i] += A[i * M + j] * x[j];
        }
    }
    printf("Vector r: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", y_host[i]);
    }
    printf("\n");
    // Imprimir resultado
    printf("Vector f: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    // Liberar memoria en el dispositivo
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    // Liberar memoria en el host
    free(A_transpose);
    free(A);
    free(x);
    free(y);
    free(y_host);
    return 0;
}



