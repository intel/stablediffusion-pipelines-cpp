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

## (Optional): Run txt-to-image pipeline using int8 quantized models

    :: Clone sd-1.5-square-quantized hugging face space
    git lfs install
    git clone https://huggingface.co/Intel/sd-1.5-square-quantized

    :: Time to run the txt-to-image example. 

    :: Start by going to the folder containing 'txt_to_img.exe' that you previously built.
    cd C:\path\to\stablediffusion-pipelines-cpp-build\intel64\Release

    :: Here is the usage of txt_to_img.exe (this can be printed using txt_to_img.exe --help):
    txt_to_image_interpolate usage:
    --prompt="some positive prompt"
    --negative_prompt="some negative prompt"
    --seed=12345
    --guidance_scale=8.0
    --num_inference_steps=20
    --model_dir="C:\Path\To\Some\Model_Dir"
    --unet_subdir="INT8" or "FP16"
    --text_encoder_device=CPU
    --unet_positive_device=CPU
    --unet_negative_device=CPU
    --vae_decoder_device=CPU
    --scheduler=["EulerDiscreteScheduler", "PNDMScheduler", "USTMScheduler"]
    --input_image=C:\SomePath\img.png
    --strength=0.75

    :: For --model_dir parameter, we will use the 'sd-1.5-square-quantized\' folder that was created from
    :: the git clone.
    :: We add additional parameter of --unet_subdir=INT8 to specify that we want the unet mode in
    :: C:\Path\To\sd-1.5-square-quantized\INT8 to be used.
    txt_to_img.exe --model_dir="C:\Path\To\sd-1.5-square-quantized" --unet_subdir="INT8" --prompt="photo of an astronaut riding a horse on mars"

    :: After generating an image, it will be displayed within a window. To generate another image using the same configuration,
    ::  click on the window and press 'c'. Otherwise if you want to quit, press 'q'.

    :: Here is a more advanced example, which makes use of negative prompt, as well as taking advantage of all accelerators -- CPU, GPU, and NPU (available on Intel Core Ultra):
    txt_to_img.exe --model_dir="C:\Path\To\sd-1.5-square-quantized\"  --unet_subdir="INT8" --prompt="professional photo of astronaut riding a horse on the moon" --negative_prompt="lowres, bad quality, monochrome" --text_encoder_device=CPU --unet_positive_device=NPU --unet_negative_device=GPU --vae_decoder_device=GPU

    :: Here is an example that uses FP16 version of unet. Note that this model is batch2, so we can't split 
    :: UNet across 2 different devices. So, we set both unet_positive_device / unet_negative_device parameters to GPU.
    txt_to_img.exe --model_dir="C:\Path\To\sd-1.5-square-quantized\"  --unet_subdir="FP16" --prompt="professional photo of astronaut riding a horse on the moon" --negative_prompt="lowres, bad quality, monochrome" --text_encoder_device=CPU --unet_positive_device=GPU --unet_negative_device=GPU --vae_decoder_device=GPU
    
    
