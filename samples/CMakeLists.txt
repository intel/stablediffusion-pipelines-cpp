
include_directories( . )

add_executable(txt_to_img txt_to_img.cpp simple_cmdline_parser.h )
if(WIN32)
   set_target_properties(txt_to_img PROPERTIES COMPILE_PDB_NAME txt_to_img)
endif()
target_link_libraries(txt_to_img PRIVATE  openvino::runtime ${OpenCV_LIBRARIES} ${ITTNOTIFY_LIB} stable_diffusion_ov)

add_executable(txt_to_img_interpolate txt_to_img_interpolate.cpp simple_cmdline_parser.h )
if(WIN32)
   set_target_properties(txt_to_img_interpolate PROPERTIES COMPILE_PDB_NAME txt_to_img)
endif()
target_link_libraries(txt_to_img_interpolate PRIVATE  openvino::runtime ${OpenCV_LIBRARIES} ${ITTNOTIFY_LIB} stable_diffusion_ov)

add_executable(simple_tokenizer_test simple_tokenizer_test.cpp  )
if(WIN32)
   set_target_properties(simple_tokenizer_test PROPERTIES COMPILE_PDB_NAME simple_tokenizer_test)
endif()
target_link_libraries(simple_tokenizer_test PRIVATE  openvino::runtime ${OpenCV_LIBRARIES} ${ITTNOTIFY_LIB} stable_diffusion_ov)

add_executable(invalid_parameter_check invalid_parameter_check.cpp simple_cmdline_parser.h )
if(WIN32)
   set_target_properties(invalid_parameter_check PROPERTIES COMPILE_PDB_NAME invalid_parameter_check)
endif()
target_link_libraries(invalid_parameter_check PRIVATE  openvino::runtime ${OpenCV_LIBRARIES} ${ITTNOTIFY_LIB} stable_diffusion_ov)



if( OVSTABLEDIFFUSION_HAS_AUDIO )
    message("Building audio-related samples... ")
	add_executable(txt_to_audio_interpolate txt_to_audio_interpolate.cpp )
	if(WIN32)
	   set_target_properties(txt_to_audio_interpolate PROPERTIES COMPILE_PDB_NAME txt_to_audio_interpolate)
	endif()
	target_link_libraries(txt_to_audio_interpolate PRIVATE  openvino::runtime ${ITTNOTIFY_LIB} stable_diffusion_audio_ov)

	add_executable(audio_to_audio audio_to_audio.cpp)
	if(WIN32)
	   set_target_properties(audio_to_audio PROPERTIES COMPILE_PDB_NAME audio_to_audio)
	endif()
	target_link_libraries(audio_to_audio PRIVATE  openvino::runtime ${ITTNOTIFY_LIB} stable_diffusion_audio_ov )

	add_executable(img_to_wav img_to_wav.cpp)
	if(WIN32)
	   set_target_properties(img_to_wav PROPERTIES COMPILE_PDB_NAME img_to_wav)
	endif()
	target_link_libraries(img_to_wav PRIVATE  openvino::runtime ${ITTNOTIFY_LIB} stable_diffusion_audio_ov ${OpenCV_LIBRARIES})
	
	add_executable(txt_to_audio txt_to_audio.cpp simple_cmdline_parser.h )
	if(WIN32)
	   set_target_properties(txt_to_audio PROPERTIES COMPILE_PDB_NAME txt_to_audio)
	endif()
	target_link_libraries(txt_to_audio PRIVATE  openvino::runtime ${OpenCV_LIBRARIES} ${ITTNOTIFY_LIB} stable_diffusion_audio_ov )

	target_compile_definitions(invalid_parameter_check PRIVATE HAS_CPP_SD_AUDIO_PIPELINE_SUPPORT )
	target_link_libraries(invalid_parameter_check PRIVATE stable_diffusion_audio_ov)
else()
    message("NOT building audio-related samples...")
endif()