#include <immintrin.h>
#include <stdio.h>


//  C = C + A*B
void dgemm(int dim1, int dim2, int dim3, double A[dim1][dim2], double Breves[dim2][dim3], double C[dim1][dim3]) {
	__m256d suma, suma_per;
	__m256d Ma, Mb, Mu;
	int particiones =  dim2 / 4;
    	int modulo = dim2 % 4;
	for (int i = 0; i < dim1; i++){
		for(int k = 0; k < dim3; k++){
		    	suma = _mm256_setzero_pd();
		    	for(int part = 0; part < particiones; part++){
		        	Ma = _mm256_loadu_pd(&A[i][part*4]);
		        	Mb = _mm256_loadu_pd(&Breves[k][part*4]);
		        	Mu = _mm256_mul_pd(Ma, Mb); //multiplico matriz por vect
		        	suma = _mm256_add_pd(suma, Mu);// guardo los elementos para ir sumandolos
		    	}
			if(modulo){ // ultimo caso dimension matriz no es multiplo de 4
				__m256i mascara = _mm256_setr_epi64x(
				    modulo >= 1 ? -1 : 0,
				    modulo >= 2 ? -1 : 0,
				    modulo >= 3 ? -1 : 0,
				    0
				);
				Ma = _mm256_maskload_pd(&A[i][dim2 - modulo], mascara);
				Mb = _mm256_maskload_pd(&Breves[k][dim2 - modulo], mascara);
				Mu = _mm256_mul_pd(Ma, Mb); 
				suma = _mm256_add_pd(suma, Mu);
			}
			// Suma los registros y deja el resultado en el primer elemento.
			suma_per = _mm256_permute2f128_pd(suma , suma , 1);// permuto para poder sumar los elementos correctamente
			suma = _mm256_add_pd(suma, suma_per);
			suma = _mm256_hadd_pd(suma, suma);
			C[i][k] += suma[0];
		}      
	}
}



int main() {
	const int dim1 = 100;  
    	const int dim2 = 100; 
    	const int dim3 = 100; 
    	double A[dim1][dim2];
    	double B[dim2][dim3];
    	double C[dim1][dim3];
    	double J[dim1][dim3];
	double Breves[dim2][dim3];
	// Inicializar A y B con valores
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
			A[i][j] = i + j + 1;  // Ejemplo: valores crecientes
		}
	}
	for (int i = 0; i < dim2; i++) {
		for (int j = 0; j < dim3; j++) {
			B[i][j] = i + j + 1;  // Ejemplo: valores crecientes
		}
	}
	// Inicializar C con ceros
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim3; j++) {
		    	C[i][j] = 0.0;
		}
	}    
	for(int i = 0; i < dim1; i++){
	    	for(int j = 0; j < dim2; j++){
	    		for(int k = 0; k < dim3; k++){
	    			J[i][k] +=  A[i][j] * B[j][k];
	    		}
	    	}
	}
	for (int i = 0; i < dim2; i++){
		for (int j = 0; j < dim3; j++){
			Breves[j][i] = B[i][j];
		}
	}
	dgemm(dim1, dim2, dim3, A, Breves, C);
	// Imprimir matriz C resultante
	printf("Matriz A resultante:\n");
    	for(int i = 0; i < dim1; i++){
        	for (int j = 0; j < dim2; j++) {
            		printf("%.1f ", A[i][j]);
       	 	}
       	 	printf("\n");
    	}
    	printf("Matriz B resultante:\n");
    	for (int i = 0; i < dim2; i++) {
        	for (int j = 0; j < dim3; j++) {
            		printf("%.1f ", B[i][j]);
       	 	}
       	 	printf("\n");
    	}
    	printf("Matriz C resultante:\n");
    	int error = 0;
    	for (int i = 0; i < dim1; i++) {
        	for (int j = 0; j < dim3; j++) {
            		if(C[i][j]-J[i][j] != 0.00){
            			printf("%.1f ", C[i][j]-J[i][j]);
            			error = 1;
            		} 
       	 	}
       	 	printf("\n");
    	}
    	if(error == 0){
    		printf("Todo ok");
    		printf("\n");
    	}else{
    		printf("Todo no ok");
    		printf("\n");
    	}
    	return 0;
}
