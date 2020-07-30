#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const int K = 100000;   // DO NOT CHANGE!!

char colon;
double x, y;

int main() {
  ifstream std("check.dat"), result("result.dat");
  for (int i = 0, _; i < K; i++) {
    std >> _ >> colon >> x;
    result >> _ >> colon >> y;
    if (abs((y - x) / x * 100000) > 1) {
      cout << i + 1 << ": should be " << x << " but read " << y << endl;
      return 1;
    }
  }
  cout << "success" << endl;
  return 0;
}
