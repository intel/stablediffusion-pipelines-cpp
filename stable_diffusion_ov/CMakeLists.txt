set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set( STABLE_DIFFUSION_OV_PUBLIC_HEADERS 
	include/cpp_stable_diffusion_ov/special_tokens_mixin.h
	include/cpp_stable_diffusion_ov/pretrained_tokenizer_base.h
	include/cpp_stable_diffusion_ov/pretrained_tokenizer.h
	include/cpp_stable_diffusion_ov/clip_tokenizer.h 
	include/cpp_stable_diffusion_ov/basic_tokenizer.h
	include/cpp_stable_diffusion_ov/openvino_text_encoder.h
	include/cpp_stable_diffusion_ov/openvino_unet.h
	include/cpp_stable_diffusion_ov/openvino_vae_decoder.h
	include/cpp_stable_diffusion_ov/openvino_vae_encoder.h
	include/cpp_stable_diffusion_ov/openvino_model_utils.h
	include/cpp_stable_diffusion_ov/model_collateral_cache.h
	include/cpp_stable_diffusion_ov/stable_diffusion_pipeline.h
	include/cpp_stable_diffusion_ov/stable_diffusion_interpolation_pipeline.h
	include/cpp_stable_diffusion_ov/callbacks.h
	include/cpp_stable_diffusion_ov/tokenization_utils.h
	include/cpp_stable_diffusion_ov/visibility.h
)

set( STABLE_DIFFUSION_OV_PRIVATE_HEADERS 
    tokenizers/unicode_const.h
	tokenizers/clip-vocab.h
	tokenizers/clip-bpe-merges.h
	pipelines/unet_loop.h
	pipelines/unet_loop_split.h
	pipelines/unet_loop_sd15_internal_blobs.h
	pipelines/unet_loop_batch2.h
	schedulers/scheduler.h
	schedulers/pndm_scheduler.h
	schedulers/ustm_scheduler.h
	schedulers/euler_discrete_scheduler.h
	schedulers/scheduler_factory.h
	include/cpp_stable_diffusion_ov/clip_const_factory.h
	utils/rng.h
)

set( STABLE_DIFFUSION_OV_SOURCES
	tokenizers/special_tokens_mixin.cpp  
	tokenizers/pretrained_tokenizer_base.cpp
	tokenizers/pretrained_tokenizer.cpp
	tokenizers/clip_tokenizer.cpp 
	
	tokenizers/basic_tokenizer.cpp
	tokenizers/clip_const_factory.cpp
	openvino_models/openvino_text_encoder.cpp
	openvino_models/openvino_unet.cpp
	openvino_models/openvino_vae_decoder.cpp
	openvino_models/openvino_vae_encoder.cpp
	openvino_models/model_collateral_cache.cpp
	pipelines/stable_diffusion_pipeline.cpp
	pipelines/stable_diffusion_interpolation_pipeline.cpp
	pipelines/unet_loop_split.cpp
	pipelines/unet_loop_sd15_internal_blobs.cpp
	pipelines/unet_loop_batch2.cpp
	schedulers/pndm_scheduler.cpp
	schedulers/ustm_scheduler.cpp
	schedulers/euler_discrete_scheduler.cpp
	schedulers/scheduler_factory.cpp
)

add_library(stable_diffusion_ov SHARED
${STABLE_DIFFUSION_OV_SOURCES}
${STABLE_DIFFUSION_OV_PUBLIC_HEADERS}
${STABLE_DIFFUSION_OV_PRIVATE_HEADERS}
)

#TEMPORARILY ADD OPENCV
message("OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")
target_link_libraries(stable_diffusion_ov PUBLIC openvino::runtime ${ITTNOTIFY_LIB} )
target_include_directories(stable_diffusion_ov PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(stable_diffusion_ov PUBLIC "include")

target_compile_features(stable_diffusion_ov PRIVATE cxx_std_17)
target_compile_definitions(stable_diffusion_ov PRIVATE IMPLEMENT_CPP_SD_OV_API )

set_target_properties(stable_diffusion_ov PROPERTIES PUBLIC_HEADER "${STABLE_DIFFUSION_OV_PUBLIC_HEADERS}")

include(GNUInstallDirs)

install(TARGETS stable_diffusion_ov
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib/static
    RUNTIME  DESTINATION bin
    RESOURCE DESTINATION bin
    PUBLIC_HEADER DESTINATION include/cpp_stable_diffusion_ov
    )

if(DEFINED ENV{LIBTORCH_ROOTDIR})
    message("LIBTORCH_ROOTDIR Env. found. Will attempt to build audio components (like riffusion pipeline) that require LibTorch.")
	set( OVSTABLEDIFFUSION_HAS_AUDIO True PARENT_SCOPE)
	
	#Libtorch is required.
	#if(DEFINED $ENV{LIBTORCH_ROOTDIR})
	#   message( FATAL_ERROR "LIBTORCH_ROOTDIR environment variable not set." )
	#endif()

	#set(LIBTORCH_ROOTDIR $ENV{LIBTORCH_ROOTDIR})
	#set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}/share/cmake/Torch")

	#find_package(Torch REQUIRED)
	
	

	set(STABLE_DIFFUSION_OV_AUDIO_PUBLIC_HEADERS 
	include/cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h 
	include/cpp_stable_diffusion_audio_ov/wav_util.h
	include/cpp_stable_diffusion_audio_ov/riffusion_audio_to_audio_pipeline.h
	include/cpp_stable_diffusion_audio_ov/stable_diffusion_audio_interpolation_pipeline.h
	include/cpp_stable_diffusion_audio_ov/visibility.h
	)

	#Separating the audio-related stuff into a different library, since it required libtorch
	add_library(stable_diffusion_audio_ov SHARED
	audio_utils/spectrogram_image_converter.cpp
	audio_utils/wav_util.cpp
	pipelines/riffusion_audio_to_audio_pipeline.cpp
    pipelines/stable_diffusion_audio_interpolation_pipeline.cpp
	${STABLE_DIFFUSION_OV_AUDIO_PUBLIC_HEADERS}
	)

	target_link_libraries(stable_diffusion_audio_ov PUBLIC openvino::runtime stable_diffusion_ov ${TORCH_LIBRARIES}  )

	target_compile_features(stable_diffusion_audio_ov PRIVATE cxx_std_17)
	target_compile_definitions(stable_diffusion_audio_ov PRIVATE IMPLEMENT_CPP_SD_AUDIO_OV_API )
	
	set(USE_OPENCV False )
	if( USE_OPENCV )
	    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc )
		target_compile_definitions(stable_diffusion_audio_ov PUBLIC USE_OPENCV)
		target_link_libraries( stable_diffusion_audio_ov PRIVATE ${OpenCV_LIBRARIES} )
	endif()

	target_include_directories(stable_diffusion_audio_ov PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

	if (MSVC)
	  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	  add_custom_command(TARGET stable_diffusion_audio_ov
						 POST_BUILD
						 COMMAND ${CMAKE_COMMAND} -E copy_if_different
						 ${TORCH_DLLS}
						 $<TARGET_FILE_DIR:stable_diffusion_audio_ov>)
	endif (MSVC)

	set_target_properties(stable_diffusion_audio_ov PROPERTIES PUBLIC_HEADER "${STABLE_DIFFUSION_OV_AUDIO_PUBLIC_HEADERS}")
	install(TARGETS stable_diffusion_audio_ov
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib/static
    RUNTIME  DESTINATION bin
    RESOURCE DESTINATION bin
    PUBLIC_HEADER DESTINATION include/cpp_stable_diffusion_audio_ov
    )

else()
    message("LIBTORCH_ROOTDIR Env. not found... Not a big deal though, we just won't' attempt to build audio components (like riffusion pipeline) that require LibTorch.")
	set( OVSTABLEDIFFUSION_HAS_AUDIO False PARENT_SCOPE)
endif()
