#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "config.h"

int main() {
  srand(time(NULL));

  freopen("input.dat", "w", stdout);
  for (int i = 0; i < m; i++) {
    double center = (double)rand() / RAND_MAX * 100;
    double max = center + 3.5;
    double min = center - 3.5;
    min = min < 0 ? 0 : min;
    for (int j = 0; j < 6; j++) {
      printf("%5.2lf%c", (double)rand() / RAND_MAX * (max - min) + min, j == 5 ? '\n' : ' ');
    }
  }
  fclose(stdout);

  freopen("K.dat", "w", stdout);
  for (int i = 0; i < K; i++)
    printf("%lf\n", (double)rand() / RAND_MAX * 0.1 + 0.95);  // 0.95 - 1.05
  fclose(stdout);
  return 0;
}
