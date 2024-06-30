#include <stdio.h>
#include <immintrin.h>  // Incluir Intel Intrinsics
static void simple_dgmv(size_t n, size_t l, float c[n], const float M[n][l], const float b[n]) {
    int mas = l % 8;
    int repeticiones =  l / 8;
    printf("mas: %d\n", mas);
    printf("repeticiones: %d\n", repeticiones);
    printf("\n");
    __m256 sum2;
    for (int i = 0; i < n; i++) {
        __m256 sum = _mm256_setzero_ps();  // Inicializar el acumulador a cero usando AVX
        // Procesar la multiplicación de la fila i de M por el vector b usando AVX
        for (int j = 0; j < repeticiones; j += 1) {  // Procesar de a 8 elementos a la vez (AVX)
	    // Cargar 8 elementos de la fila i de M y el vector b usando AVX
	    __m256 m = _mm256_loadu_ps(&M[i][j*8]);
	    __m256 v = _mm256_loadu_ps(&b[j*8]); 
	    // Multiplicar y acumular en el registro sum
	    sum = _mm256_add_ps(sum, _mm256_mul_ps(m, v));
	    printf("suma00");
	    printf("\n");
	}
	printf("suma11");
	printf("\n");
	for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum[k]);
 	}
 	printf("\n");
	if(mas){
		__m256i mask = _mm256_setr_epi32(
				mas >= 1 ? -1 : 0,
				mas >= 2 ? -1 : 0,
				mas >= 3 ? -1 : 0,
				mas >= 4 ? -1 : 0,
				mas >= 5 ? -1 : 0,
				mas >= 6 ? -1 : 0,
				mas >= 7 ? -1 : 0,
				0
               );
        	__m256 m, v;
	    	// Cargar los elementos restantes de la fila i de M y el vector b usando la máscara
	    	m = _mm256_maskload_ps(&M[i][l - mas], mask);
	    	v = _mm256_maskload_ps(&b[n - mas], mask);
	    	// Multiplicar y acumular en el registro sum
	    	sum = _mm256_add_ps(sum, _mm256_mul_ps(m, v));
	    	// Sumar horizontalmente los elementos del registro sum para obtener la suma final
	}
	// Almacenar el resultado en el vector c
	printf("suma");
	printf("\n");
	for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum[k]);
 	}
 	printf("\n");
 	printf("suma2");
	printf("\n");
 	sum2 = _mm256_permute2f128_ps(sum , sum , 1); //Se realizan operaciones para sumar horizontalmente los elementos del registro
 	for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum2[k]);
 	}
 	printf("\n");
 	printf("suma");
	printf("\n");
	sum = _mm256_hadd_ps(sum, sum2);
	for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum[k]);
 	}
 	printf("\n");
 	printf("suma");
	printf("\n");
        sum = _mm256_hadd_ps(sum, sum);
        for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum[k]);
 	}
 	printf("\n");
 	printf("suma");
	printf("\n");
	sum = _mm256_hadd_ps(sum, sum);	
	for (int k = 0;k<8;k++){
 	 	printf("%.1f ", sum[k]);
 	}
 	printf("\n");
        // Sumar horizontalmente los elementos del registro sum para obtener la suma final 
        // Almacenar el resultado en el vector c
        c[i] += sum[0];
    }
}

int main() {
    const size_t n = 14;  // Tamaño de la matriz y vectores
    const size_t l = 14;
    float M[n][l];
    float c[n];
    float b[n];
    float Mprueba[n][l];
    float cprueba[n];
    float bprueba[n];
    
    // Declarar y definir la matriz M y los vectores c y b
    printf("Resultado de M[][]:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < l; j++) {
            M[i][j] = i * n + j + 1;  // valor creciente
            Mprueba[i][j] = i * n + j + 1;  // valor creciente
            printf("%.1f ", M[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Inicializar el vector b
    for (int i = 0; i < l; i++) {
        b[i] = i + 1;  // valor creciente
        bprueba[i] = i + 1 ;
    }

    // Inicializar el vector c con ceros
    for (int i = 0; i < n; i++) {
        c[i] = 0.0f;
        cprueba[i] = 0.0f;
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < l; j++) {
            cprueba[i] += Mprueba[i][j] * bprueba[j];
        }
    }


    simple_dgmv(n, l, c, M, b);
    printf("Resultado de b[]:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", b[i]);
    }
    
    printf("\n");
    printf("Resultado de c[]:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", c[i]);
    }
    printf("\n");
    printf("Resultado de cprueba[]:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", cprueba[i]);
    }
    printf("\n");
    // Imprimir el resultado
    printf("Resultado de la multiplicacion de la matriz por el vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", c[i]-cprueba[i]);
    }
    printf("\n");

    return 0;
}

