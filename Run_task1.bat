@echo off
setlocal

cd Task1

if not exist build mkdir build
cd build

cmake -Wno-dev ..

cmake --build .

cd Debug

echo Running Original Matrix Multiplication...
matrix_mul_original.exe

echo Running Optimized Matrix Multiplication...
matrix_mul_optimized.exe

pause

endlocal
