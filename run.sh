make
rm result.dat
mpirun -ppn 4 ./logVS
./checker
