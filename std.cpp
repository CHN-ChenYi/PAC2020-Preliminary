/*
 * logDataVSPrior is a function to calculate
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <omp.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

typedef chrono::high_resolution_clock Clock;

#include "config.h"

double logDataVSPrior(const double* dat_real, const double* dat_imag,
                      const double* pri_real, const double* pri_imag,
                      const double* ctf, const double* sigRcp, const int num,
                      const double disturb0);

int main(int argc, char* argv[]) {
  double *dat_real = new double[m], *dat_imag = new double[m];
  double *pri_real = new double[m], *pri_imag = new double[m];
  double* ctf = new double[m];
  double* sigRcp = new double[m];
  double* disturb = new double[K];

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

  ofstream fout;
  fout.open("check.dat");
  if (!fout.is_open()) {
    cout << "Error opening file for check" << endl;
    exit(1);
  }

  for (unsigned int t = 0; t < K; t++) {
    double result = logDataVSPrior(dat_real, dat_imag, pri_real, pri_imag, ctf,
                                   sigRcp, m, disturb[t]);
    fout << t + 1 << ": " << result << endl;
  }
  fout.close();

  auto endTime = Clock::now();

  auto compTime =
      chrono::duration_cast<chrono::microseconds>(endTime - startTime);
  cout << "Computing time=" << compTime.count() << " microseconds" << endl;

  delete[] dat_real;
  delete[] dat_imag;
  delete[] pri_real;
  delete[] pri_imag;

  delete[] ctf;
  delete[] sigRcp;
  delete[] disturb;
  return EXIT_SUCCESS;
}

inline double pow_2(const double& x) { return x * x; }

double logDataVSPrior(const double* dat_real, const double* dat_imag,
                      const double* pri_real, const double* pri_imag,
                      const double* ctf, const double* sigRcp, const int num,
                      const double disturb0) {
  double result = 0.0;
#pragma omp parallel for reduction(+ : result) schedule(static)
  for (int i = 0; i < num; i++) {
    result += (pow_2(dat_real[i] - disturb0 * ctf[i] * pri_real[i]) +
               pow_2(dat_imag[i] - disturb0 * ctf[i] * pri_imag[i])) *
              sigRcp[i];
  }
  return result;
}
