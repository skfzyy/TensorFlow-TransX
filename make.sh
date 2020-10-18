scriptPath=`dirname $0`
g++ ${scriptPath}/init.cpp -o ${scriptPath}/init.so -fPIC -shared -pthread -O3 -march=native
g++ ${scriptPath}/test.cpp -o ${scriptPath}/test.so -fPIC -shared -pthread -O3 -march=native
