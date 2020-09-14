#include "cuda_utils.h"
#include "linalg.h"
#include <stdio.h> 
#include "cublas_v2.h"

#ifdef CUDA
extern cublasHandle_t cublas_handle;
#endif


__global__ void dgemm_naive(const double *A, const double *B, double *C,
                            const int M, const int N, const int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double sum = 0.;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void dgemm_ta_naive(const double *A, const double *B, double *C,
			       const int M, const int N, const int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double sum = 0;
    for (int k = 0; k < K; k++)
      sum += A[k * M + row] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

__global__ void dgemm_tb_naive(const double *A, const double *B, const double *C, double *D,
			       const int M, const int N, const int K) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double sum = 0;
    for (int k = 0; k < K; k++)
      sum += A[row * K + k] * B[col * K + k];
    D[row * N + col] = sum + C[row * N + col];
  }
}


//  Optimized matrix multiply kernels using shared memory.
// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
__global__ void dgemm_optimized(const double *A, const double *B, double *C,
				const int M, const int N, const int K) {

  double CValue = 0;

  extern __shared__ double shared[];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int dim = blockDim.x;

  int Row = blockIdx.y * dim + ty;
  int Col = blockIdx.x * dim + tx;
  
  double* As = (double*)shared;
  double* Bs = (double*)&shared[dim*dim];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (dim + ACols -1)/dim; k++) {

    if (k*dim + tx < ACols && Row < ARows)
      As[ty * dim + tx] = A[Row * ACols + k * dim+ tx];
    else
      As[ty * dim + tx] = 0.0;

    if (k * dim + ty < BRows && Col < BCols)
      Bs[ty * dim + tx] = B[(k * dim * BCols) + (ty * BCols) + Col];
    else
      Bs[ty * dim + tx] = 0.0;

    __syncthreads();

    for (int n = 0; n < dim; n++) {
        CValue += As[ty * dim + n] * Bs[n * dim + tx];
    }
    __syncthreads();
   }

   if (Row < CRows && Col < CCols) {
     C[Row*CCols + Col] = CValue;
  }
}


// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
__global__ void dgemm_ta_optimized(const double *A, const double *B, double *C,
				   const int M, const int N, const int K) {

  double CValue = 0;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int dim = blockDim.x;

  int Row = blockIdx.y * dim + ty;
  int Col = blockIdx.x * dim + tx;

  extern __shared__ double shared[];

  double* As = (double*)shared;
  double* Bs = (double*)&shared[dim*dim];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (dim + ACols -1)/dim; k++) {
    if (k * dim + tx < ACols && Row < ARows)
      As[ty * dim + tx] = A[(k * dim + tx)*ARows + Row];
    else
      As[ty * dim + tx] = 0.0;

    if (k * dim + ty < BRows && Col < BCols)
      Bs[ty * dim + tx] = B[(k * dim + ty)*BCols + Col];                            
    else
      Bs[ty * dim + tx] = 0.0;

  __syncthreads();

   for (int n = 0; n < dim; n++)
      CValue += As[ty * dim + n] * Bs[n * dim + tx];

    __syncthreads();
   }

   if (Row < CRows && Col < CCols) {
     C[Row*CCols + Col] = CValue;
  }
}


__global__ void dgemm_tb_optimized(const double *A, const double *B, const double *C, double *D, const size_t M,
                               const size_t K, const size_t N) {

  double DValue = 0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int dim = blockDim.x;

  int Row = blockIdx.y * dim + ty;
  int Col = blockIdx.x * dim + tx;

  extern __shared__ double shared[];
  double* As = (double*)shared;
  double* Bs = (double*)&shared[dim*dim];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (dim + ACols -1)/dim; k++) {

    if (k*dim + tx < ACols && Row < ARows)
      As[ty * dim + tx] = A[Row*ACols + k*dim + tx];
    else
      As[ty * dim + tx] = 0.0;

    if (k*dim + ty < BRows && Col < BCols)
      Bs[ty * dim + tx] = B[k*dim + Col*BRows + ty];  
    else
      Bs[ty * dim + tx] = 0.0;

    __syncthreads();

    for (int n = 0; n < dim; n++)
      DValue += As[ty * dim + n] * Bs[n * dim + tx];

    __syncthreads();
  }
  
  if (Row < CRows && Col < CCols) {
    D[Row*CCols + Col] = DValue + C[Row*CCols + Col];
  }
}


// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifndef CUBLAS
  int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_naive<<<grid, block>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());


#elif defined(_GPU_GEMM_OPT)
  
  int TILE_DIM = 16;

  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x, (M + dimBlock.y - 1)/dimBlock.y);
  size_t shmem_size = 2 * TILE_DIM * TILE_DIM * sizeof(double);

  dgemm_optimized<<<dimGrid, dimBlock, shmem_size>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());


#endif
#else
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         A * B = (A^T)^T * (B^T)^T = A'^T * B'^T = (B' * A')^T
 
const double alpha(1.0);
const double beta(0.0);

cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, N,
            A, K,
            &beta,
            C, N);

#endif
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta_gpu(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifndef CUBLAS
  int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_ta_naive<<<grid, block>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)

  int TILE_DIM = 16;
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);                                                                                   
  dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x, (M + dimBlock.y - 1)/dimBlock.y);
  size_t shmem_size = 2 * TILE_DIM * TILE_DIM * sizeof(double);

  dgemm_ta_optimized<<<dimGrid, dimBlock, shmem_size>>>(A, B, C, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());

#endif
#else
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         A^T * B = A^T * (B^T)^T = A' * B'^T = (B'*A'^T)^T
//		M is KxM, B is KxN, C is MxN.
/*
 *  FILLME: Use cublasDgemm()
 */

const double alpha(1.0);
const double beta(0.0);                                                                          

cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K,
            &alpha,
            B, N,
            A, M,
            &beta,
            C, N);

#endif
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb_gpu(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K) {
#ifndef CUBLAS
  int BLOCK_SIZE = 16;
#if defined(_GPU_GEMM_NAIVE)
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dgemm_tb_naive<<<grid, block>>>(A, B, C, D, M, N, K);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(_GPU_GEMM_OPT)
 
  int TILE_DIM = 16;
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x, (M + dimBlock.y - 1)/dimBlock.y);
  size_t shmem_size = 2 * TILE_DIM * TILE_DIM * sizeof(double);

  dgemm_tb_optimized<<<dimGrid, dimBlock, shmem_size>>>(A, B, C, D, M, K, N);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());


#endif
#else
// D = A * B^T
// Matrices are stored in row-major order, but cuBLAS assumes column-major
// order. We want to compute:
//         C = A * B^T = (A^T)^T * B^T  = A'^T * B' = (B'^T * A')^T
//		A is MxK, B is NxK, C is MxN, D is MxN

const double alpha(1.0);
const double beta(0.0);

// D = A * B'
cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
           N, M, K,
           &alpha,
           B, K,
           A, K,
           &beta,
           D, N);

// D = C + D
cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M,
            &alpha,
            D, N,
            &alpha,
            C, N,
            D, N);                                                                                    

#endif
}
