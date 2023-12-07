# How to build for Windows :hammer:

## Dependencies
Here are some of the dependencies that you need to grab, and a quick description of how to set up environment (from cmd.exe shell)
* **CMake & Visual Studio** (MSVC 2019 or 2022, community edition is fine)
* **OpenVINO™** - You can use public version from [here](https://github.com/openvinotoolkit/openvino/releases/tag/2023.1.0). Setup your cmd.exe shell environment by running setupvars.bat:  
    ```
    call "C:\path\to\w_openvino_toolkit_windows_xxxx\setupvars.bat"
    ```
* **OpenCV** - Only a dependency for the openvino-stable-diffusion-cpp samples (to read/write images from disk, display images, etc.). You can find pre-packages Windows releases [here](https://github.com/opencv/opencv/releases). We currently use 4.8.1 with no issues, it's recommended that you use that.
   ```
   set OpenCV_DIR=C:\path\toopencv\build
   set Path=%OpenCV_DIR%\x64\vc16\bin;%Path%
   ```
* **(*Optional*) Libtorch (C++ distribution of pytorch)** - This is a dependency for the audio utilities (like spectrogram-to-wav, wav-to-spectrogram), and audio pipelines.  We are currently using this version: [libtorch-win-shared-with-deps-2.1.1+cpu.zip](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.1%2Bcpu.zip). After extracting the package, setup environment like this:
    ```
    set LIBTORCH_ROOTDIR=C:\path\to\libtorch-shared-with-deps-2.1.1+cpu\libtorch
    set Path=%LIBTORCH_ROOTDIR%\lib;%Path%
    ```
    
    Note: If Libtorch is not installed, the build will simply **not build** the audio-related utilities and pipelines.

## Build Commands (assuming cmd.exe shell with above environment set)

    :: clone it
    git clone https://github.com/intel/stablediffusion-pipelines-cpp.git
  
    :: create build folder
    mkdir stablediffusion-pipelines-cpp-build
    cd stablediffusion-pipelines-cpp-build
  
    :: run cmake
    cmake ../stablediffusion-pipelines-cpp

    :: Build it:
    cmake --build . --config Release

    :: After this step, you'll see the built collateral in stablediffusion-pipelines-cpp-build\intel64\Release
  
    :: (Optional): Install built DLL's & headers into a custom directory
    ::  Usually only used if you want to then use this in another project.
    cmake --install . --config Release --prefix <path_to>/installed
