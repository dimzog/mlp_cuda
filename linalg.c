#include "linalg.h"
#include "cblas.h"


// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifdef CBLAS
/*
 *  FILLME: Use cblas_dgemm()
 */

 cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
             M, N, K,
             1.0,
             A, K,
             B, N,
             0.0,
             C, N);

#else
  int i, j, k;
  double sum;
  /*
   * FILLME: Parallelize the code.
   */
  int OMP_NUM_THREADS;
  #pragma omp parallel for collapse(2) private(k, sum) num_threads(OMP_NUM_THREADS)
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      sum = 0.;
      for (k = 0; k < K; k++)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
#endif
}

// Computes C = A'*B, where A is a K by M matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
void dgemm_ta(const double *A, const double *B, double *C, const int M, const int N, const int K) {
#ifdef CBLAS
/*
 *  FILLME: Use cblas_dgemm()
 */

 cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
             M, N, K,
             1.0,
             A, M,
             B, N,
             0.0,
             C, N);

#else
  int i, j, k;
  double sum;
  /*
   * FILLME: Parallelize the code.
   */
  int OMP_NUM_THREADS;
  #pragma omp parallel for collapse(2) private(k, sum) num_threads(OMP_NUM_THREADS)
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      sum = 0.;
      for (k = 0; k < K; k++)
        sum += A[k * M + i] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
#endif
}

// Computes D = A*B'+C, where A is a M by K matrix, B is a N by K matrix, C and D are M by N matrices.
// Matrices are stored in row-major order.
void dgemm_tb(const double *A, const double *B, const double *C, double *D, const int M, const int N, const int K) {
#ifdef CBLAS
/*
 *  FILLME: Use cblas_dgemm()
 */
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
             M, N, K,
             1.0,
             A, K,
             B, K,
             1.0,
             C, N);

// Copy elements of C to D
for (int i=0; i<N*M; i++){ D[i] += C[i]; }


#else
  int i, j, k;
  double sum;
  /*
   * FILLME: Parallelize the code.
   */
  int OMP_NUM_THREADS;
  #pragma omp parallel for collapse(2) private(k, sum) num_threads(OMP_NUM_THREADS)
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      sum = 0.;
      for (k = 0; k < K; k++)
        sum += A[i * K + k] * B[j * K + k];
      D[i * N + j] = sum + C[i * N + j];
    }
  }
#endif
}

void hadamard2D(double *out, const double *in1, const double *in2, const int M, const int N) {
  int i, j;
  double res;
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
      res = in1[i * N + j] * in2[i * N + j];
      out[i * N + j] = res;
    }
}

void sumRows(double *out, const double *in, const int M, const int N) {
  memset(out, 0, N * sizeof(double));
  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++)
      out[j] += in[i * N + j];
  }
}

void gradientb(double *out, const double *in, const int M, const int N, const double lr) {
  int i, j;
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      out[i * N + j] -= (lr)*in[j];
}

void gradientW(double *out, const double *in, const int M, const int N, const double lr) {
  int i, j;
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      out[i * N + j] -= (lr)*in[i * N + j];
}
