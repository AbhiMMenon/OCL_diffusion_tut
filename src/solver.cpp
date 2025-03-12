#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <cmath>

#define H(IX  ,  IY)         h[(IY)*width + (IX)]
#define NX 2000
#define NY 2000

#define L_DOM 1. // meter
#define H_DOM 1. // meter

void printMat(const float *f, const size_t width, const size_t height){
    if(width*height < 100)
    for(size_t jx = 0; jx < height; jx++){
        for(size_t ix = 0; ix < width; ix++){
            printf("%f ", f[jx*width + ix]);
        }
        printf("\n");
    }
}

/**
 * Writes a matrix h in gnuplot binary format
 *
 * @param   fout        file pointer, file openened with binary write
 * @param   m           field to be written, contiguous buffer
 * @param   xsize 
 * @param   ysize 
 * @param   rt          x positions
 * @param   ct          y positions
 */

int fwrite_matrix( FILE *fout, float *h, int xsize, int ysize, float *rt, float *ct)
{
    int j;
    int status;
    float length = xsize;

    if ((status = fwrite((char *) &length, sizeof(float), 1, fout)) != 1) {
	fprintf(stderr, "fwrite 1 returned %d\n", status);
	return (0);
    }
    fwrite((char *) ct, sizeof(float), ysize, fout);
    for (j = 0; j < ysize; j++){
	fwrite((char *) &rt[j], sizeof(float), 1, fout);

	fwrite(h+j*xsize, sizeof(float), xsize, fout);
    }
    return (1);
}



int main(int argc, char** argv)
{

    // -- parse arguments
   int c1; 
   size_t n=1; float d=1.;
   while((c1=getopt(argc, argv, "n:d:"))!=-1){
       switch(c1){
           case 'n':
               n = atoi(optarg);
               break;
           case 'd':
               d = atof(optarg);
               break;
            default:
               printf("Usage: ./diffusion -n {iterations} -d {diffusivity} \n");
               return(1);

       }

   }
   printf("Iterations = %ld, diffCoeff = %f \n",n,d);

    size_t width =  NX;
    size_t height = NY;
    std::vector<float> h,x,y;
    FILE *fp; 
    printf("Width, height =  %ld, %ld\n", width, height);
    h.resize(width*height,0.);
    x.resize(width,0.);
    y.resize(height,0.);

    // y positions
    float yStep = H_DOM/height;
    float xStep = L_DOM/width;
    y[0] = yStep*0.5;
    for(size_t jx = 1; jx < height; jx++){ 
        y[jx] = y[jx-1] + yStep;
    }
    x[0] = xStep*0.5;
    for(size_t ix = 1; ix < width; ix++){ 
        x[ix] = x[ix-1] + xStep;
    }
    
    // --- iniital field
    for(size_t jx = 0; jx < height; jx++){
        for(size_t ix = 1; ix < width; ix++){ 
            float r = (0.5-x[ix])* (0.5-x[ix]) + (0.5-y[jx])*(0.5-y[jx]);
            if (r< 0.15 && r > 0.05)
                H(ix, jx) = 1.0;
        }
    }

    printMat(h.data(),width, height);
   
   fp = fopen(INIT_DAT, "wb"); 
   // check
   if(!fp){
       fprintf(stderr,  "Could not open ouput file:%s \n", OUT_DAT);
       fclose(fp);
       return 1;
   }
   fwrite_matrix(fp, h.data(), width, height, x.data(), y.data());
   fclose(fp);


  
//-- OpenCL boilerplate 

  cl_int error;
  cl_platform_id platform_id;
  cl_uint nmb_platforms;
  if ( clGetPlatformIDs(1, &platform_id, &nmb_platforms) != CL_SUCCESS ) {
    fprintf(stderr, "cannot get platform\n" );
    return 1;
  }

  cl_device_id device_id;
  cl_uint nmb_devices;
  if ( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &nmb_devices) != CL_SUCCESS ) {
    fprintf(stderr, "cannot get device\n" );
    return 1;
  }

  cl_context context;
  cl_context_properties properties[] =
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) platform_id,
    0
  };
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create context\n");
    return 1;
  }

  cl_command_queue command_queue;
  command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create command queue\n");
    return 1;
  }
	
// --- read file kernel.cl, load as program
  char *opencl_program_src;
  {
    FILE *clfp = fopen(CL_SOURCE, "r");
    if ( clfp == NULL ) {
      fprintf(stderr, "could not load cl source code\n");
      return 1;
    }
    fseek(clfp, 0, SEEK_END);
    int clfsz = ftell(clfp);
    fseek(clfp, 0, SEEK_SET);
    opencl_program_src = (char*) malloc((clfsz+1)*sizeof(char));
    fread(opencl_program_src, sizeof(char), clfsz, clfp);
    opencl_program_src[clfsz] = 0;
    fclose(clfp);
  }

  cl_program program;
  size_t src_len = strlen(opencl_program_src);
  program = clCreateProgramWithSource(
                context, 1, (const char **) &opencl_program_src, (const size_t*) &src_len, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create program\n");
    return 1;
  }

  free(opencl_program_src);
  
  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot build program. log:\n");
      
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    char *log = (char*) malloc(log_size*sizeof(char));
    if ( log == NULL ) {
      fprintf(stderr, "could not allocate memory\n");
      return 1;
    }

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "%s\n", log );
    free(log);

    return 1;
  }

// ----		Kernels,buffers and solver	---- //
  
  cl_kernel diffuse_kernel_locMem = clCreateKernel(program, "diffuseNew", &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create kernel diffuse_kernel_2\n");
    return 1;
  }


  const int sz = width*height;
  const int local_sy = BLOCK_SIZE;
  const int local_sx = BLOCK_SIZE;


  // -- device memory
  cl_mem buffer_h;
  buffer_h = clCreateBuffer(context, CL_MEM_READ_WRITE, sz*sizeof(float), NULL, &error);
  if ( error != CL_SUCCESS ) {
    fprintf(stderr, "cannot create buffer a\n");
    return 1;
  }
  

  // -- copy from host to device
  if ( clEnqueueWriteBuffer(command_queue, buffer_h, CL_TRUE, 0, sz*sizeof(float), h.data(), 0, NULL, NULL)
       != CL_SUCCESS ) {
    fprintf(stderr, "cannot enqueue write of buffer a\n");
    return 1;
  }


  clSetKernelArg(diffuse_kernel_locMem, 0, sizeof(cl_mem), &buffer_h);
  clSetKernelArg(diffuse_kernel_locMem, 1, sizeof(int), &width);
  clSetKernelArg(diffuse_kernel_locMem, 2, sizeof(int), &height);
  clSetKernelArg(diffuse_kernel_locMem, 3, sizeof(int), &d);
  
  for(size_t nx = 0; nx < n; nx++){
    
      const size_t global[2] = {height,width};
      const size_t local[2] = {local_sy,local_sx};

      if ( clEnqueueNDRangeKernel(command_queue, diffuse_kernel_locMem, 2, NULL, (const size_t*) &global, (const size_t*) &local, 0, NULL, NULL)
           != CL_SUCCESS ) {
        fprintf(stderr, "Enque kernel failed!!\n");
        return 1;
      }
  
  }


  if ( clEnqueueReadBuffer(command_queue,buffer_h, CL_TRUE, 0, sz*sizeof(float), h.data(), 0, NULL, NULL)
      != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue read of buffer \n");
    return 1;
  }

  if ( clFinish(command_queue) != CL_SUCCESS ) {
    fprintf(stderr, "cannot finish queue\n");
    return 1;
  }
  

   printf("\n");
   printMat(h.data(),width, height);
   fp = fopen(OUT_DAT, "wb"); 
   if(!fp){
       fprintf(stderr,  "Could not open ouput file:%s \n", OUT_DAT);
       fclose(fp);
       return 1;
   }
   fwrite_matrix(fp, h.data(), width, height, x.data(), y.data());
   fclose(fp);
  
  clReleaseMemObject(buffer_h);

  clReleaseProgram(program);
  clReleaseKernel(diffuse_kernel_locMem);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);



  return 0;
}
