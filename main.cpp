/*
 * logDataVSPrior is a function to calculate
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <immintrin.h>
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

typedef chrono::high_resolution_clock Clock;

const int m = 1638400;  // DO NOT CHANGE!!
const int K = 100000;   // DO NOT CHANGE!!

int mpi_id, mpi_size, block_size, m_;
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
MPI_Request mpi_request[K];
float *dat_real, *dat_imag, *pri_real, *pri_imag, *ctf, *sigRcp, *disturb, *ans;

template <typename T>
inline T pow_2(const T &x) {
  return x * x;
}

inline float logDataVSPrior(const float *dat_real, const float *dat_imag,
                            const float *pri_real, const float *pri_imag,
                            const float *ctf, const float *sigRcp,
                            const int num, const float disturb0) {
  float result = 0.0;
  union {
    float tmp_result[8];
    __m256 total;
  };
  total = _mm256_setzero_ps();
#pragma omp parallel shared(total)
  {
    __m256 _disturb0 = _mm256_broadcast_ss(&disturb0);
#pragma omp declare reduction(addps:__m256         \
                              : omp_out += omp_in) \
    initializer(omp_priv = _mm256_setzero_ps())
#pragma omp for schedule(static) reduction(addps : total)
    for (int i = 0; i < num; i += 8) {
      // do not use over 16 registers in total or the processor writes back to
      // L1 cache.
      __m256 _dat_real0 = _mm256_load_ps(dat_real + i);
      __m256 _pri_real0 = _mm256_load_ps(pri_real + i);
      __m256 _dat_imag0 = _mm256_load_ps(dat_imag + i);
      __m256 _pri_imag0 = _mm256_load_ps(pri_imag + i);
      __m256 _ctf0 = _mm256_load_ps(ctf + i);
      __m256 _sigRcp0 = _mm256_load_ps(sigRcp + i);

      total += (pow_2(_dat_real0 - _disturb0 * _ctf0 * _pri_real0) +
                pow_2(_dat_imag0 - _disturb0 * _ctf0 * _pri_imag0)) *
               _sigRcp0;
    }
  }
  for (int i = 0; i < 8; i++) {
    result += tmp_result[i];
  }
  return result;
}

inline void Read();
inline void Compute(float *tmp_ans[]);
inline void Print();

int main(int argc, char *argv[]) {
  /*******
   * init
   * ****/

  std::ios::sync_with_stdio(false);
  std::cin.tie(0);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  block_size = m / mpi_size;
  full_size = (mpi_size - 1) * block_size, remainder_size = m - full_size;
  if (mpi_id == mpi_size - 1)
    m_ = remainder_size;
  else
    m_ = block_size;

  dat_real = (float *)_mm_malloc(sizeof(float) * m, 32);
  dat_imag = (float *)_mm_malloc(sizeof(float) * m, 32);
  pri_real = (float *)_mm_malloc(sizeof(float) * m, 32);
  pri_imag = (float *)_mm_malloc(sizeof(float) * m, 32);
  ctf = (float *)_mm_malloc(sizeof(float) * m, 32);
  sigRcp = (float *)_mm_malloc(sizeof(float) * m, 32);
  disturb = (float *)_mm_malloc(sizeof(float) * K, 32);
  ans = (float *)_mm_malloc(sizeof(float) * K, 32);
  float *tmp_ans[mpi_size];
  for (int i = 1; i < mpi_size; i++)
    tmp_ans[i] = (float *)_mm_malloc(sizeof(float) * K, 32);

  Read();

  /***************************
   * main computation is here
   * ************************/

  MPI_Barrier(MPI_COMM_WORLD);
  decltype(Clock::now()) startTime;
  if (!mpi_id) startTime = Clock::now();

  Compute(tmp_ans);

  if (!mpi_id) {
    Print();

    auto endTime = Clock::now();
    auto compTime =
        chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;
  }

  /***********
   * finalize
   * ********/

  MPI_Finalize();

  _mm_free(dat_real);
  _mm_free(dat_imag);
  _mm_free(pri_real);
  _mm_free(pri_imag);

  _mm_free(ctf);
  _mm_free(sigRcp);
  _mm_free(disturb);
  _mm_free(ans);
  for (int i = 1; i < mpi_size; i++) _mm_free(tmp_ans[i]);

  return EXIT_SUCCESS;
}

inline void Read() {
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
    if (i == block_size * mpi_id) break;
  }
  i = 0;
  while (!fin.eof()) {
    fin >> dat_real[i] >> dat_imag[i] >> pri_real[i] >> pri_imag[i] >> ctf[i] >>
        sigRcp[i];
    i++;
    if (i == m_) break;
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
}

const int OMP_M_BLOCK_SIZE = 500000;
// const int OMP_K_BLOCK_SIZE = 100000;
inline void Compute(float *tmp_ans[]) {
  for (unsigned int m_start = 0, m_len, m_flag = true; m_flag;
       m_start += OMP_M_BLOCK_SIZE) {
    if (m_start + 2 * OMP_M_BLOCK_SIZE < m_) {
      m_len = OMP_M_BLOCK_SIZE;
    } else {
      m_len = m_ - m_start;
      m_flag = false;
    }
    for (unsigned int t = 0; t < K; t++)
      ans[t] =
          logDataVSPrior(dat_real + m_start, dat_imag + m_start,
                         pri_real + m_start, pri_imag + m_start, ctf + m_start,
                         sigRcp + m_start, m_len, disturb[t]);
    // for (unsigned int k_start = 0, k_end, k_flag = true; k_flag;
    //      k_start += OMP_K_BLOCK_SIZE) {
    //   if (k_start + 2 * OMP_K_BLOCK_SIZE < K) {
    //     k_end = k_start + OMP_K_BLOCK_SIZE;
    //   } else {
    //     k_end = K;
    //     k_flag = false;
    //   }
    //   if (!m_start) {
    //     for (unsigned int t = k_start; t < k_end; t++)
    //       ans[t] = logDataVSPrior(dat_real + m_start, dat_imag + m_start,
    //                               pri_real + m_start, pri_imag + m_start,
    //                               ctf + m_start, sigRcp + m_start, m_len,
    //                               disturb[t]);
    //   } else {
    //     for (unsigned int t = k_start; t < k_end; t++)
    //       ans[t] += logDataVSPrior(dat_real + m_start, dat_imag + m_start,
    //                                pri_real + m_start, pri_imag + m_start,
    //                                ctf + m_start, sigRcp + m_start, m_len,
    //                                disturb[t]);
    //   }
    // }
  }

  if (mpi_id) {  // send ans to node 0
    MPI_Send(ans, K, MPI_FLOAT, 0, kAns, MPI_COMM_WORLD);
  } else {  // collect ans from other nodes and print ans
    for (int i = 1; i < mpi_size; i++)
      MPI_Irecv(tmp_ans[i], K, MPI_FLOAT, i, kAns, MPI_COMM_WORLD,
                &mpi_request[i]);
    for (int i = 1; i < mpi_size; i++) {
      MPI_Wait(&mpi_request[i], &mpi_status);
#pragma omp parallel for schedule(static)
      for (int t = 0; t < K; t++) ans[t] += tmp_ans[i][t];
    }
  }
}

inline void Print() {
  ofstream fout;

  fout.open("result.dat");
  if (!fout.is_open()) {
    cout << "Error opening file for result" << endl;
    exit(1);
  }

  const unsigned int length = 8*4194304;
  static char buffer[length];

  // for (int t = 0; t < K; t++) fout << t + 1 << ": "8 << ans[t] << '\n';

  int offset = 0;

  for (int t = 0; t < K; t++) {
    offset += sprintf(buffer + offset, "%d: %11.5e\n", t + 1, ans[t]);
  }

/* #pragma omp parallel for schedule(static)
  for (int t = 0; t < 9; ++t) {
    sprintf(buffer + (t << 4) - (t << 1), "%d:%11.5e", t + 1, ans[t]);
    buffer[(t << 4) - (t << 1) + 13] = '\n';
  }
#pragma omp parallel for schedule(static)
  for (int t = 9; t < 99; ++t) {
    sprintf(buffer + -9 + (t << 4) - t, "%d:%11.5e", t + 1, ans[t]);
    buffer[5 + (t << 4) - t] = '\n';
  }
#pragma omp parallel for schedule(static)
  for (int t = 99; t < 999; ++t) {
    sprintf(buffer + -108 + (t << 4), "%d:%11.5e", t + 1, ans[t]);
    buffer[-93 + (t << 4)] = '\n';
  }
#pragma omp parallel for schedule(static)
  for (int t = 999; t < 9999; ++t) {
    sprintf(buffer + -1107 + (t << 4) + t, "%d:%11.5e", t + 1, ans[t]);
    buffer[-1091 + (t << 4) + t] = '\n';
  }
#pragma omp parallel for schedule(static)
  for (int t = 9999; t < 99999; ++t) {
    sprintf(buffer + -11106 + (t << 4) + (t << 1), "%d:%11.5e", t + 1, ans[t]);
    buffer[-11089 + (t << 4) + (t << 1)] = '\n';
  }

  sprintf(buffer + 1788876, "%d:%11.5e\n", 100000, ans[99999]);

  int len = strlen(buffer); */
  fout.write(buffer, offset);

  fout.close();
}
