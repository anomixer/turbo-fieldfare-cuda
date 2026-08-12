@echo off
setlocal EnableExtensions

echo === NVIDIA driver and GPU ===
nvidia-smi

echo.
echo === CUDA compiler ===
nvcc --version

echo.
echo === Expected Ada configuration ===
echo Configure Ada and L4 builds with -DTF_CUDA_ARCHITECTURES=89-real

echo.
echo === System memory ===
systeminfo | findstr /C:"Total Physical Memory"

echo.
echo === Free disk space ===
wmic logicaldisk get caption,freespace,size

endlocal
