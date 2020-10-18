scriptPath=`dirname $0`
g++ ${scriptPath}/init.cpp -o init.so -fPIC -shared -pthread -O3 -march=native
g++ ${scriptPath}/test.cpp -o test.so -fPIC -shared -pthread -O3 -march=native
