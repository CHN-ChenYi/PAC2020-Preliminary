g++ ./gen.cpp -o ./gen
icpc ./std.cpp -o ./std -fopenmp -O3 -xHost
g++ ./checker.cpp -o ./checker
echo Compiled
./gen
echo Gened
./std
