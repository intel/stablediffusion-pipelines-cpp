#include <iostream>
#include "simple_cmdline_parser.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "cpp_stable_diffusion_ov/stable_diffusion_interpolation_pipeline.h"
#ifdef HAS_CPP_SD_AUDIO_PIPELINE_SUPPORT
#include "cpp_stable_diffusion_audio_ov/riffusion_audio_to_audio_pipeline.h"
#include "cpp_stable_diffusion_audio_ov/stable_diffusion_audio_interpolation_pipeline.h"
#endif

void print_usage()
{
	std::cout << "invalid_parameter usage: " << std::endl;
	std::cout << "--model_dir=\"C:\\Path\\To\\Some\\Model_Dir\" " << std::endl;
}

bool CheckInvalidParameters_StableDiffusion(std::optional<std::string> model_dir)
{
	cpp_stable_diffusion_ov::StableDiffusionPipeline sd_pipeline(*model_dir, {},
		"CPU", "CPU", "CPU", "CPU", "CPU");

	// Catch invalid inference steps
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto image_buf = sd_pipeline("some positive prompt",
			"some negative prompt",
			-1, //num inference steps
			"EulerDiscreteScheduler",
			12345, //seed
			7.f, //guidance scale
			true, // give us BGR back
			input_image_params);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusion: Expected exception caught: " << error.what()  << std::endl;
	}

	// Catch invalid scheduler selection
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto image_buf = sd_pipeline("some positive prompt",
			"some negative prompt",
			10, //num inference steps
			"SomeNonexistentScheduler",
			12345, //seed
			7.f, //guidance scale
			true, // give us BGR back
			input_image_params);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusion: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (negative)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = -1.;  //bogus strength
		input_image_params = params;
		auto image_buf = sd_pipeline("some positive prompt",
			"some negative prompt",
			10, //num inference steps
			"EulerDiscreteScheduler",
			12345, //seed
			7.f, //guidance scale
			true, // give us BGR back
			input_image_params);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusion: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (> 1.0)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 2.;  //bogus strength
		input_image_params = params;
		auto image_buf = sd_pipeline("some positive prompt",
			"some negative prompt",
			10, //num inference steps
			"EulerDiscreteScheduler",
			12345, //seed
			7.f, //guidance scale
			true, // give us BGR back
			input_image_params);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusion: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameters, forgot to set image buffer
	try
	{

		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;

		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 0.5f;  
		input_image_params = params;
		auto image_buf = sd_pipeline("some positive prompt",
			"some negative prompt",
			10, //num inference steps
			"EulerDiscreteScheduler",
			12345, //seed
			7.f, //guidance scale
			true, // give us BGR back
			input_image_params);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusion: Expected exception caught: " << error.what() << std::endl;
	}


	return true;
}

bool CheckInvalidParameters_StableDiffusionInterpolation(std::optional<std::string> model_dir)
{
	cpp_stable_diffusion_ov::StableDiffusionInterpolationPipeline sd_pipeline(*model_dir, {},
		"CPU", "CPU", "CPU", "CPU", "CPU");


	// Catch invalid inference steps
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			7.f, //guidance scale start
			8.f, //guidance scale stop
			0, //num inference steps
			5, //interpolation steps
			"EulerDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid scheduler selection
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			7.f, //guidance scale start
			8.f, //guidance scale stop
			10, //num inference steps
			5, //interpolation steps
			"NewtonDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid interpolation steps
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			7.f, //guidance scale start
			8.f, //guidance scale stop
			10, //num inference steps
			-1, //interpolation steps
			"EulerDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (negative)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = -1.;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			7.f, //guidance scale start
			8.f, //guidance scale stop
			10, //num inference steps
			2, //interpolation steps
			"EulerDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (> 1.0)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 1.1;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			8.f, //guidance scale start
			8.f, //guidance scale stop
			10, //num inference steps
			2, //interpolation steps
			"EulerDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameters, forgot to set image buffer
	try
	{
		
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;

		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 0.5f;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			12345, //seed start
			23456, //seed end
			8.f, //guidance scale start
			8.f, //guidance scale stop
			10, //num inference steps
			2, //interpolation steps
			"EulerDiscreteScheduler",
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	return true;
}


bool CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha(std::optional<std::string> model_dir)
{
	cpp_stable_diffusion_ov::StableDiffusionInterpolationPipeline sd_pipeline(*model_dir, {},
		"CPU", "CPU", "CPU", "CPU", "CPU");


	// Catch invalid inference steps
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			0.5f, //alpha
			-11, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid scheduler selection
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			0.5f, //alpha
			10, //num inference steps
			"SomethingWrong", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid alpha (negative)
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			-1.f, //alpha
			10, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	//Catch invalid alpha (>1.f)
	try
	{
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			2.f, //alpha
			10, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (negative)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = -1.;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			0.7, //alpha
			10, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameter strength (>1.0)
	try
	{
		auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
		params.image_buffer = buf;
		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 3.f;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			0.7, //alpha
			10, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid image parameters, forgot to set image buffer
	try
	{

		std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
		cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;

		params.isBGR = true;
		params.isNHWC = true;
		params.strength = 0.5f;  //bogus strength
		input_image_params = params;
		auto out_img_vec = sd_pipeline.run_single_alpha(
			"some start prompt",
			"some end prompt",
			"some negative prompt",
			0.7, //alpha
			10, //num inference steps
			"EulerDiscreteScheduler", //scheduler
			12345, //seed start
			23456, //seed end
			8.f, // guidance scale
			true, // give us BGR back
			input_image_params //input image, strength, etc.
		);

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha: Expected exception caught: " << error.what() << std::endl;
	}

	return true;
}

#ifdef HAS_CPP_SD_AUDIO_PIPELINE_SUPPORT
bool CheckInvalidParameters_StableDiffusionAudioInterpolation(std::optional<std::string> model_dir)
{
	cpp_stable_diffusion_ov::StableDiffusionAudioInterpolationPipeline audio_interp_pipeline(*model_dir, {},
		"CPU", "CPU", "CPU", "CPU", "CPU");

#if 0
	auto out_samples = audio_interp_pipeline(true, //give stereo?
		"some start prompt",
		"some negative prompt",
		"some end prompt",
		{}, //negative end -- not used.
		12345, //seed start
		23456, //seed end
		0.5f, // start strength 
		0.6f, // end strength
		7.f, //guidance scale start,
		8.f, //guidance scale end,
		10, //num inference steps
		5, //num interpolation steps,
		2, //num output segments
		"og_beat", //seed image
		1.f,
		"EulerDiscreteScheduler");
#endif
	// Catch invalid start strength (negative)
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			-1.f, // start strength 
			0.6f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid start strength (>1)
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.5f, // start strength 
			0.6f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid end strength (<1)
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			-1100.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid end strength (>1)
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			101.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid inference steps
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			0.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			-1, //num inference steps
			5, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid interpolation steps
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			0.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			-1, //num interpolation steps,
			2, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid output segments
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			0.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			6, //num output segments
			"og_beat", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid seed image
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			0.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			4, //num output segments
			"mc_jammin", //seed image
			1.f,
			"EulerDiscreteScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	// Catch invalid scheduler
	try
	{
		auto out_samples = audio_interp_pipeline(true, //give stereo?
			"some start prompt",
			"some negative prompt",
			"some end prompt",
			{}, //negative end -- not used.
			12345, //seed start
			23456, //seed end
			1.0f, // start strength 
			0.f, // end strength
			7.f, //guidance scale start,
			8.f, //guidance scale end,
			10, //num inference steps
			5, //num interpolation steps,
			5, //num output segments
			"og_beat", //seed image
			1.f,
			"FFMScheduler");
		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudioInterpolation: Expected exception caught: " << error.what() << std::endl;
	}

	return true;
}

bool CheckInvalidParameters_StableDiffusionAudio2Audio(std::optional<std::string> model_dir)
{
	cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline pipeline(*model_dir, {},
		"CPU", "CPU", "CPU", "CPU", "CPU");

	int available_samples = 44100 * 10;
	std::vector< float > samples(available_samples);

	

	// Catch L ptr null
	try
	{
		auto out_samples = pipeline(nullptr,
			nullptr,
			44100 * 2, //samples to remix
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// catch samples to remix is 0
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			0, //samples to riffuse
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// total number of samples < samples to process
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			44100,
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// number of inference steps is < 0
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			-1, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// invalid scheduler
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"BuffaloDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// invalid strength ( <0)
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			-100.f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// invalid strength ( > 1)
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			1.5f,
			0.2f); //0.2 is overlap of individual riffused segments that are cross-faded.

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// invalid cross-fade (< 0)
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			-1.f); //cross-fade 

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	// invalid cross-fade (> 1)
	try
	{
		auto out_samples = pipeline(samples.data(),
			nullptr,
			44100 * 2, //samples to process
			samples.size(),
			"some prompt",
			{},
			20, //num inference steps
			"EulerDiscreteScheduler",
			12345,
			7.5,
			0.5f,
			1.5f); //cross-fade 

		std::cout << "Invalid parameter not caught." << std::endl;
		return false;
	}
	catch (const std::invalid_argument& error) {
		std::cout << "CheckInvalidParameters_StableDiffusionAudio2Audio: Expected exception caught: " << error.what() << std::endl;
	}

	return true;

}

#endif

int main(int argc, char* argv[])
{
	try
	{
		SimpleCmdLineParser cmdline_parser(argc, argv);
		if (cmdline_parser.is_help_needed())
		{
			print_usage();
			return -1;
		}

		std::optional<std::string> model_dir;
		model_dir = cmdline_parser.get_value_for_key("model_dir");
		if (!model_dir)
		{
			std::cout << "Error! --model_dir argument is required" << std::endl;
			print_usage();
			return 1;
		}

		if (!CheckInvalidParameters_StableDiffusion(model_dir))
		{
			std::cout << "Error: Invalid parameter not caught for SD pipeline" << std::endl;
			return -1;
		}

		if (!CheckInvalidParameters_StableDiffusionInterpolation(model_dir))
		{
			std::cout << "Error: Invalid parameter not caught for SD Interpolation pipeline" << std::endl;
			return -1;
		}

		if (!CheckInvalidParameters_StableDiffusionInterpolationSingleAlpha(model_dir))
		{
			std::cout << "Error: Invalid parameter not caught for SD Interpolation Single Alpha pipeline" << std::endl;
			return -1;
		}

#ifdef HAS_CPP_SD_AUDIO_PIPELINE_SUPPORT
		if (!CheckInvalidParameters_StableDiffusionAudioInterpolation(model_dir))
		{
			std::cout << "Error: Invalid parameter not caught for SD Audio Interpolation pipeline" << std::endl;
			return -1;
		}


		if (!CheckInvalidParameters_StableDiffusionAudio2Audio(model_dir))
		{
			std::cout << "Error: Invalid parameter not caught for SD Audio2Audio Interpolation pipeline" << std::endl;
			return -1;
		}
#endif


		std::cout << "All invalid parameter checks passed." << std::endl;
	}
	catch (const std::exception& error) {
		std::cout << "in invalid parameter check routine: exception: " << error.what() << std::endl;
		return -1;
	}

	return 0;
}