/*
 * logDataVSPrior is a function to calculate
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "mpi.h"

using namespace std;

typedef chrono::high_resolution_clock Clock;

const int m = 1638400;  // DO NOT CHANGE!!
const int K = 100000;   // DO NOT CHANGE!!
int mpi_id, mpi_size, block_size;
size_t full_size, remainder_size;
enum MPI_TAG {
  kDatReal,
  kDatImag,
  kPriReal,
  kPriImag,
  kCtf,
  kSigRcp,
  kDisturb,
  kAns
};
MPI_Status mpi_status;
float *dat_real, *dat_imag, *pri_real, *pri_imag, *ctf, *sigRcp, *disturb, *ans;

inline float pow_2(const float &x) { return x * x; }

inline float logDataVSPrior(const float *dat_real, const float *dat_imag,
                            const float *pri_real, const float *pri_imag,
                            const float *ctf, const float *sigRcp,
                            const int num, const float disturb0) {
  float result = 0.0;
#pragma omp parallel for reduction(+ : result) schedule(static)
  for (int i = 0; i < num; i++) {
    result += (pow_2(dat_real[i] - disturb0 * ctf[i] * pri_real[i]) +
               pow_2(dat_imag[i] - disturb0 * ctf[i] * pri_imag[i])) *
              sigRcp[i];
  }
  return result;
}

inline void Server() {
  MPI_Request mpi_request[K];
  float *tmp_ans[mpi_size];
  for (int i = 1; i < mpi_size; i++) tmp_ans[i] = new float[K];

  /***************************
   * Read data from input.dat
   * *************************/
  ifstream fin;

  fin.open("input.dat");
  if (!fin.is_open()) {
    cout << "Error opening file input.dat" << endl;
    exit(1);
  }
  int i = 0;
  while (!fin.eof()) {
    fin >> dat_real[i] >> dat_imag[i] >> pri_real[i] >> pri_imag[i] >> ctf[i] >>
        sigRcp[i];
    i++;
    if (i == m) break;
  }
  fin.close();

  fin.open("K.dat");
  if (!fin.is_open()) {
    cout << "Error opening file K.dat" << endl;
    exit(1);
  }
  i = 0;
  while (!fin.eof()) {
    fin >> disturb[i];
    i++;
    if (i == K) break;
  }
  fin.close();

  /***************************
   * main computation is here
   * ************************/
  auto startTime = Clock::now();

  for (int i = 1; i < mpi_size; i++)
    MPI_Send(disturb, K, MPI_FLOAT, i, kDisturb, MPI_COMM_WORLD);
#define SEND(POINTER, TAG)                                                    \
  for (int i = 1; i < mpi_size - 1; i++)                                      \
    MPI_Send(POINTER + i * block_size, block_size, MPI_FLOAT, i, TAG,         \
             MPI_COMM_WORLD);                                                 \
  MPI_Send(POINTER + full_size, remainder_size, MPI_FLOAT, mpi_size - 1, TAG, \
           MPI_COMM_WORLD);
  SEND(dat_real, kDatReal);
  SEND(dat_imag, kDatImag);
  SEND(pri_real, kPriReal);
  SEND(pri_imag, kPriImag);
  SEND(ctf, kCtf);
  SEND(sigRcp, kSigRcp);
#undef SEND

  ofstream fout;

  /* const unsigned int length = 1048576;
  char buffer[length];
  fout.rdbuf()->pubsetbuf(buffer, length); */

  fout.open("result.dat");
  if (!fout.is_open()) {
    cout << "Error opening file for result" << endl;
    exit(1);
  }

  for (unsigned int t = 0; t < K; t++)
    ans[t] = logDataVSPrior(dat_real, dat_imag, pri_real, pri_imag, ctf, sigRcp,
                            block_size, disturb[t]);

  for (int i = 1; i < mpi_size; i++)
    MPI_Irecv(tmp_ans[i], K, MPI_FLOAT, i, kAns, MPI_COMM_WORLD,
              &mpi_request[i]);
  for (int i = 1; i < mpi_size; i++) {
    MPI_Wait(&mpi_request[i], &mpi_status);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < K; t++) ans[t] += tmp_ans[i][t];
  }

  const unsigned int length = 4194304;
  static char buffer[length];

  // for (int t = 0; t < K; t++) fout << t + 1 << ": " << ans[t] << '\n';

  int offset = 0;

  for (int t = 0; t < K; t++) {
    offset += sprintf(buffer + offset, "%d: %6e\n", t + 1, ans[t]);
  }

  // int len = strlen(buffer);
  fout.write(buffer, offset);

  fout.close();

  auto endTime = Clock::now();

  auto compTime =
      chrono::duration_cast<chrono::microseconds>(endTime - startTime);
  cout << "Computing time=" << compTime.count() << " microseconds" << endl;

  for (int i = 1; i < mpi_size; i++) delete[] tmp_ans[i];
}

inline void Client() {
  int m_;
  if (mpi_id == mpi_size - 1)
    m_ = remainder_size;
  else
    m_ = block_size;
  MPI_Recv(disturb, K, MPI_FLOAT, 0, kDisturb, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(dat_real, m_, MPI_FLOAT, 0, kDatReal, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(dat_imag, m_, MPI_FLOAT, 0, kDatImag, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(pri_real, m_, MPI_FLOAT, 0, kPriReal, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(pri_imag, m_, MPI_FLOAT, 0, kPriImag, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(ctf, m_, MPI_FLOAT, 0, kCtf, MPI_COMM_WORLD, &mpi_status);
  MPI_Recv(sigRcp, m_, MPI_FLOAT, 0, kSigRcp, MPI_COMM_WORLD, &mpi_status);

  for (unsigned int t = 0; t < K; t++)
    ans[t] = logDataVSPrior(dat_real, dat_imag, pri_real, pri_imag, ctf, sigRcp,
                            m_, disturb[t]);

  MPI_Send(ans, K, MPI_FLOAT, 0, kAns, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  dat_real = new float[m];
  dat_imag = new float[m];
  pri_real = new float[m];
  pri_imag = new float[m];
  ctf = new float[m];
  sigRcp = new float[m];
  disturb = new float[K];
  ans = new float[K];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  block_size = m / mpi_size;
  full_size = (mpi_size - 1) * block_size, remainder_size = m - full_size;

  if (!mpi_id)
    Server();
  else
    Client();

  MPI_Finalize();

  delete[] dat_real;
  delete[] dat_imag;
  delete[] pri_real;
  delete[] pri_imag;

  delete[] ctf;
  delete[] sigRcp;
  delete[] disturb;
  delete[] ans;
  return EXIT_SUCCESS;
}
