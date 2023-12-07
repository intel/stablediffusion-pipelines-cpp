# How to build for Linux (Ubuntu 22.04)

Hi! The following is the process that we use when building for Linux. These instructions are for Ubuntu 22.04, so you may need to adapt them slightly for other Linux distros.

## Dependencies
Here are some of the dependencies that you need to grab. If applicable, I'll also give the cmd's to set up your environment here.
* Build Essentials (GCC & CMake)
```
sudo apt install build-essential
```
* OpenVINO - You can use public version from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/linux/l_openvino_toolkit_ubuntu22_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz)
```
# Extract it
tar xvf l_openvino_toolkit_ubuntu22_2023.2.0.13089.cfd42bd2cb0_x86_64.tgz 

#install dependencies
cd l_openvino_toolkit_ubuntu22_2023.2.0.13089.cfd42bd2cb0_x86_64/install_dependencies/
sudo -E ./install_openvino_dependencies.sh
cd ..

# setup env
source setupvars.sh
```
* OpenCV - Only a dependency for the sample applications (to read/write images from disk, display images, etc.). You can install like this:
```
sudo apt install libopencv-dev
```

* Libtorch (C++ distribution of pytorch)- This is a dependency for the audio utilities (like spectrogram-to-wav, wav-to-spectrogram). We are currently using this version: [libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip ](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcpu.zip). Setup environment like this:
```
unzip libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip
export LIBTORCH_ROOTDIR=/path/to/libtorch
```

## Build it
```
# clone it
git clone https://github.com/intel/stablediffusion-pipelines-cpp.git

#create build folder
mkdir stablediffusion-pipelines-cpp-build
cd stablediffusion-pipelines-cpp-build

# Run cmake
cmake ../stablediffusion-pipelines-cpp

# Build it
make -j 8

# Install it
cmake --install . --config Release --prefix ./installed

# Set environment variable that Audacity module will use to find this component.
export CPP_STABLE_DIFFUSION_OV_ROOTDIR=/path/to/stablediffusion-pipelines-cpp-build/installed
export LD_LIBRARY_PATH=${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib:$LD_LIBRARY_PATH
```

