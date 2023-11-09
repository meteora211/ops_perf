cmake -S ./ -B `pwd`/build -DCMAKE_BUILD_TYPE=Debug
cmake --build `pwd`/build --target install --config Debug
