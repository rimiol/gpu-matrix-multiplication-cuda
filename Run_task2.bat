@echo off
setlocal

cd Task2

if not exist build mkdir build
cd build

cmake -Wno-dev ..

cmake --build .

cd Debug

echo Running Modified Matrix Multiplication...
matrix_mul_block.exe

pause

endlocal