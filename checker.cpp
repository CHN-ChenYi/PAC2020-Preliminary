#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const int K = 100000;   // DO NOT CHANGE!!

char colon;
double x, y;
double max_de, de;

int main() {
  ifstream std("check.dat"), result("result.dat");
  for (int i = 0, _; i < K; i++) {
    std >> _ >> colon >> x;
    result >> _ >> colon >> y;
    de = abs((y - x) / x * 100000);
    max_de = max_de > de ? max_de : de;
    if (de > 1) {
      cout << i + 1 << ": should be " << x << " but read " << y << '(' << de << ')' << endl;
      return 1;
    }
  }
  cout << "success, max de: " << max_de << endl;
  return 0;
}
