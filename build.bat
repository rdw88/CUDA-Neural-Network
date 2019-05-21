@echo off

if "%1" == "" (
    echo Provide a build action (either 'test' for an executable with tests or 'app' for the shared library)
)

if "%1" == "test" (
    cd network/
    nvcc -O3 -Iinclude/ -o bin/test.exe NeuralNetwork.cu GPU.cu Util.cu Activation.cu Test.cu
    cd ..
)

if "%1" == "app" (
    cd network/
    nvcc -O3 -shared -Iinclude/ -o bin/ann.dll NeuralNetwork.cu GPU.cu Util.cu Extern.cu Activation.cu
    cd ..
)