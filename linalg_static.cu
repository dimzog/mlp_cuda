#include "cuda_utils.h"
#include "linalg.h"
#include <stdio.h> 
#include "cublas_v2.h"

#ifdef CUDA
extern cublasHandle_t cublas_handle;
#endif

const int TILE_DIM = 16;


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
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = blockIdx.y*TILE_DIM + ty;
  int Col = blockIdx.x*TILE_DIM + tx;
  
  __shared__ double As[TILE_DIM][TILE_DIM];
  __shared__ double Bs[TILE_DIM][TILE_DIM];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (TILE_DIM + ACols -1)/TILE_DIM; k++) {
    if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
      As[ty][tx] = A[Row*ACols + k*TILE_DIM + tx];
    else
      As[ty][tx] = 0.0;
    if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
      Bs[ty][tx] = B[(k*TILE_DIM + ty)*BCols + Col];
    else
      Bs[ty][tx] = 0.0;

    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n)
        CValue += As[ty][n] * Bs[n][tx];
      __syncthreads();
   }
   int index_c = 0;
   if (Row < CRows && Col < CCols) {
     index_c = (blockIdx.y * blockDim.y + ty) * CCols + (blockIdx.x * blockDim.x + tx);
     C[index_c] = CValue;
  }
}


// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
__global__ void dgemm_ta_optimized(const double *A, const double *B, double *C,
				   const int M, const int N, const int K) {

  double CValue = 0;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = blockIdx.y*TILE_DIM + ty;
  int Col = blockIdx.x*TILE_DIM + tx;
  __shared__ double As[TILE_DIM][TILE_DIM];
  __shared__ double Bs[TILE_DIM][TILE_DIM];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (TILE_DIM + ACols -1)/TILE_DIM; k++) {
    if (k*TILE_DIM + tx < ACols && Row < ARows)
      As[ty][tx] = A[(k*TILE_DIM + tx)*ARows + Row];
    else
      As[ty][tx] = 0.0;

    if (k*TILE_DIM + ty < BRows && Col < BCols)
      Bs[ty][tx] = B[(k*TILE_DIM + ty)*BCols + Col];                            
    else
      Bs[ty][tx] = 0.0;

  __syncthreads();

   for (int n = 0; n < TILE_DIM; ++n)
      CValue += As[ty][n] * Bs[n][tx];

    __syncthreads();
   }
   int index_c = 0;
   if (Row < CRows && Col < CCols) {
     index_c = (blockIdx.y * blockDim.y + ty) * CCols + (blockIdx.x * blockDim.x) + tx;
     C[index_c] = CValue;
  }
}


__global__ void dgemm_tb_optimized(const double *A, const double *B, const double *C, double *D, const size_t M,
                               const size_t K, const size_t N) {

  double DValue = 0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = blockIdx.y*TILE_DIM + ty;
  int Col = blockIdx.x*TILE_DIM + tx;

  __shared__ double As[TILE_DIM][TILE_DIM];
  __shared__ double Bs[TILE_DIM][TILE_DIM];

  int ARows = M; int ACols = K;
  int BRows = K; int BCols = N;
  int CRows = M; int CCols = N;

  for (int k = 0; k < (TILE_DIM + ACols -1)/TILE_DIM; k++) {

    if (k*TILE_DIM + tx < ACols && Row < ARows)
      As[ty][tx] = A[Row*ACols + k*TILE_DIM + tx];
    else
      As[ty][tx] = 0.0;

    if (k*TILE_DIM + ty < BRows && Col < BCols)
      Bs[ty][tx] = B[k*TILE_DIM + Col*BRows + ty];  
    else
      Bs[ty][tx] = 0.0;

    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n)
      DValue += As[ty][n] * Bs[n][tx];

    __syncthreads();
  }
  
  int index_c = 0;
  if (Row < CRows && Col < CCols) {
    index_c = (blockIdx.y * blockDim.y + ty) * CCols + (blockIdx.x * blockDim.x) + tx;
    D[index_c] = DValue + C[index_c];
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
  /*
   *  FILLME: Set up the blocks, grid and the shared memory size.
   */

  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid;

  dimGrid.x = (N + dimBlock.x - 1)/dimBlock.x;
  dimGrid.y = (M + dimBlock.y - 1)/dimBlock.y;

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
  /*
   *  FILLME: Set up the blocks, grid and the shared memory size.
   */

  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);                                                                                   
  dim3 dimGrid;                                                                                                             
  dimGrid.x = (N + dimBlock.x - 1)/dimBlock.x;                                                                     
  dimGrid.y = (M + dimBlock.y - 1)/dimBlock.y;

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
 
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid;                                                           
  dimGrid.x = (N + dimBlock.x - 1)/dimBlock.x;
  dimGrid.y = (M + dimBlock.y - 1)/dimBlock.y;

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
