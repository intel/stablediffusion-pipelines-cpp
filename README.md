# C++ Stable Diffusion Pipelines using OpenVINO™ 

A set of Stable Diffusion pipelines (and related utilities) ported entirely to C++ (from python), with easy-to-use API’s and a focus on minimal third-party dependencies. The core stable-diffusion libraries built by this project only have dependencies on OpenVINO™.

## Build Instructions :hammer:
  - [Windows Build Instructions](doc/build_doc/windows/README.md)  
  - [Linux Build Instructions](doc/build_doc/linux/README.md)

## Contribution :handshake:
  Your contributions are welcome and valued, no matter how big or small. Feel free to submit a pull-request!

## Credits & Acknowledgements :pray:
* This project was inspired in many ways by Georgi Gerganov's low-dependency style CPP projects such as [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* Hugging Face diffusers project (https://github.com/huggingface/diffusers): The core stable-diffusion pipeline, the tokenizers, the schedulers, etc. that were ported to C++ came from the python code found in HF diffusers project.
* The audio generation pipelines, and other pipelines that implement prompt / latent space interpolation were ported from the open source Riffusion project: https://github.com/riffusion/riffusion.git
* The audio utilities (spec-to-wav, wav-to-spec) were ported to C++ by referencing the equivelant python routines found within the PyTorch audio project (https://github.com/pytorch/audio)
