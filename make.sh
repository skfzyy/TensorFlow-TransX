scriptPath=`dirname $0`
echo ${scriptPath}
g++ ${scriptPath}/init.cpp -o ${scriptPath}/init.so -fPIC -shared -pthread -O3 -march=native
echo "compile init.so success"
g++ ${scriptPath}/test.cpp -o ${scriptPath}/test.so -fPIC -shared -pthread -O3 -march=native
echo "compile test.so success"
