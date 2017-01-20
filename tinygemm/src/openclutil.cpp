#include <string>
#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/kernelsnips.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/defaultoutpath.hpp>
#include  <CL/cl.h>


namespace tinygemm{
namespace openclutil{

void set_platform_etc(cl_platform_id & platform, cl_uint & num_platforms, cl_context & context, cl_device_id & device_id_to_use, outputwriting::OutputWriter & mowri){
  
  /* Get the platform(s) */
  clGetPlatformIDs(1, &platform, &num_platforms);
  cl_int status;
  
  /* Create context */
  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
  cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;
  context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &status);
  if (status != CL_SUCCESS){
    std::string errm("There's been an error in set_platform_etc in openclutil.cpp: the return status of clCreateContextFromType is not CL_SUCCESS. The status returned is ");
    errm += status;
    errm + ". ";
    throw tinygemm_error(errm);
  }
  
  /* Query the number of devices */
  int deviceListSize;
  clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(int), &deviceListSize, nullptr);
  
  if (deviceListSize == 0){
    throw tinygemm_error("There are no devices detected. \nSpecifically, using clGetContextInfo with CL_CONTEX_NUM_DEVICES as the flag returns 0. \nThis error is being thrown from set_platform_etc in openclutil.cpp. Please have a look...");
  }

  
  std::vector<cl_device_id> devices(deviceListSize);
  
  /* Get the devices */
  clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceListSize*sizeof(cl_device_id), devices.data(), nullptr);
  char deviceName[100];
  cl_uint max_compute_units;  
  
  /* Go through the devices and see how many compute units each has, storing the best along the way */
  cl_uint max_max_compute_units = 0;
  char bestDeviceName[100];
  cl_device_id bestDeviceId; 
  
  std::string device_compute_unit_count_string = "The following devices were detected: \n";

  for (int i = 0; i < deviceListSize; ++i){
    /* get name and number of compute units of the device whose id is devices[i] */
    

    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, nullptr);
    
    if (max_compute_units > max_max_compute_units){
      max_max_compute_units = max_compute_units;
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(bestDeviceName), bestDeviceName, nullptr);
      bestDeviceId = devices[i];
    }

    device_compute_unit_count_string += deviceName;
    device_compute_unit_count_string += ", which as ";
    device_compute_unit_count_string += std::to_string(max_compute_units);
    device_compute_unit_count_string +=  " compute units\n";
  }
  
  
  bool only_good_hardware = true;
  if (only_good_hardware == true && max_max_compute_units < 40){
    std::string errm = device_compute_unit_count_string;
    errm += "As this is less than 64, an error is being thrown. \nIf you wish to use a device with fewer than 64 CUs, please make changes here (in openclutil.cpp)";
    throw tinygemm_error(errm);
  }
  
  
  else{
    mowri << "Will use device " << bestDeviceName << ", which has " << max_max_compute_units << " CUs. \nTo use a different device, consider modifying set_platform_etc in openclutil.cpp (or write custom OpenCL boilerplate)." << Endl;
    device_id_to_use = bestDeviceId;
  }
  
  

}



void set_program_and_kernel(cl_program & program, cl_kernel & kernel, std::string & kernel_function_name, const cl_context & context, const cl_device_id & device_id_to_use, const std::string & kernel_string){
  cl_int ret;
  kernel_function_name = kernelutil::get_kernel_function_name(kernel_string);
  
  //char *source;
  //size_t sourceSize;
  //FILE *fp = fopen(filename.c_str(), "r");
  //fseek(fp, 0, SEEK_END);
  //sourceSize = ftell(fp);
  //fseek(fp , 0, SEEK_SET);
  //source = (char *)malloc(sourceSize * sizeof(char));
  //fread(source, 1, sourceSize, fp);
  //fclose(fp);
  //program = clCreateProgramWithSource(context, 1, (const char **)&source, &sourceSize, &ret);
  //free(source);
  
  auto kernel_cstr = kernel_string.c_str();
  auto kernel_string_size =  kernel_string.size();
  program = clCreateProgramWithSource(context, 1, &kernel_cstr, &kernel_string_size , &ret);
  if (ret != 0){
    throw tinygemm_error("Error in clCreateProgramWithSource (in openclutil.cpp)");
  }
  
  /* To generate isa code, add this :   ``` -save-temps= + defpaths::isacodedir + "/"  '''    to the following string  */
  std::string buildOptions_11 = "-cl-std=CL2.0";
  auto buildOptions = buildOptions_11.c_str();
  ret = clBuildProgram(program, 1, &device_id_to_use, buildOptions, NULL, NULL); 
  
  
  
  /* writing binary : from http://www.cs.bris.ac.uk/home/simonm/montblanc/AdvancedOpenCL_full.pdf */
  if (false){
    /* TODOEVENTUALLY This will break if more than one device used */
    size_t size; 
    /* the size of the binary is ridiculous. 60 MB!! */
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, &size, NULL);
    std::vector<unsigned char> v_binary (size + 10);
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, size, v_binary.data(), NULL); 
    //std::string filename_bin(filename);
    std::string filename_bin = "bladibla_bin";
    FILE *filebinary = fopen(filename_bin.c_str(), "w");
    fwrite(v_binary.data(), 1, size, filebinary);
    /* the binary file has been written to filename_bin */ 
    throw tinygemm_error("go look at created binary file!");
    /* done */
  }
  
  if (ret != 0){
    char buffer[10240];
    clGetProgramBuildInfo(program, device_id_to_use, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
    fprintf(stderr, "CL Compilation failed:\n%s", buffer);
    throw tinygemm_error("Error in clBuildProgram");
  }
  
  kernel = clCreateKernel(program, kernel_function_name.c_str(), &ret);  
  if (ret != CL_SUCCESS){
    throw tinygemm_error("Error in clCreateKernel");
  }
}  
  
  
}} // end namespace

