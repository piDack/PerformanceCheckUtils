#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    int m = 2;
    int n = 2;
    int k = 2;
    int lda = 2;
    int ldb = 2;
    int ldc = 2;
    int batch_count = 2;

    /*
     *   A = | 1.0 | 2.0 | 5.0 | 6.0 |
     *       | 3.0 | 4.0 | 7.0 | 8.0 |
     *
     *   B = | 5.0 | 6.0 |  9.0 | 10.0 |
     *       | 7.0 | 8.0 | 11.0 | 12.0 |
     */

    std::vector<std::vector<data_type>> A_array;
    std::vector<std::vector<data_type>> B_array;
    std::vector<std::vector<data_type>> C_array(batch_count, std::vector<data_type>(m * n));

    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    std::vector<data_type *> d_A(batch_count, nullptr);
    std::vector<data_type *> d_B(batch_count, nullptr);
    std::vector<data_type *> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    for(int i=0;i<batch_count;i++){
        std::vector<data_type> tmpA;
        gen_random_matrix_2_d<data_type>(&tmpA,m,n,&lda,&d_A[i]);
        std::vector<data_type> tmpB;
        gen_random_matrix_2_d<data_type>(&tmpB,n,k,&ldb,&d_B[i]);
        A_array.push_back(tmpA);
        B_array.push_back(tmpB);
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * C_array[i].size()));
    }
    /* step 2: copy data to device */

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_count));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
                                    d_B_array, ldb, &beta, d_C_array, ldc, batch_count));

    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i], sizeof(data_type) * C_array[i].size(),
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
     *       | 43.0 | 50.0 | 151.0 | 166.0 |
     */

    printf("C[0]\n");
    print_matrix(m, n, C_array[0].data(), ldc);
    printf("=====\n");

    printf("C[1]\n");
    print_matrix(m, n, C_array[1].data(), ldc);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
