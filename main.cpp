#include <iostream>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include "SFML/Graphics.hpp"

//resolution of the window
const int width = 1280;
const int height = 720;

void generate_julia_set(sf::VertexArray& vertexarray, float* results, int precision)
{
    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int iterations = results[i * 1280 + j];
            if (iterations < precision / 4.0f)
            {
                vertexarray[i*width + j].position = sf::Vector2f(j, i);
                sf::Color color(iterations * 255.0f / (precision / 4.0f), 0, 0);
                vertexarray[i*width + j].color = color;
            }
            else if (iterations < precision / 3.0f)
            {
                vertexarray[i*width + j].position = sf::Vector2f(j, i);
                sf::Color color(0, iterations * 255.0f / (precision / 3.0f), 0);
                vertexarray[i*width + j].color = color;
            }
            else if (iterations < precision / 2.0f)
            {
                vertexarray[i*width + j].position = sf::Vector2f(j, i);
                sf::Color color(0, 0, iterations * 255.0f / (precision / 2.0f));
                vertexarray[i*width + j].color = color;
            }
            else if (iterations < precision)
            {
                vertexarray[i*width + j].position = sf::Vector2f(j, i);
                sf::Color color(0, iterations * 255.0f / precision, iterations * 255.0f / precision);
                vertexarray[i*width + j].color = color;
            }
        }
    }
}

#define DATA_SIZE (1280 * 720)

const char *KernelSource = "\n" \
"__kernel void Calculate_Julia_Set(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count, int pixel_shift_x, int pixel_shift_y, int precision, float zoom, int x, int y)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = i % 1280; int k = i / 1280;                                 \n" \
"   float c_real = ((float)x) / zoom  - pixel_shift_x / 1280.0f;                    \n" \
"   float c_imag = ((float)y) / zoom  - pixel_shift_y / 720.0f;                    \n" \
"   float z_real = ((float)j) / zoom  - pixel_shift_x / 1280.0f;                                               \n" \
"   float z_imag = ((float)k) / zoom  - pixel_shift_y / 720.0f; int iterations = 0;                             \n" \
"   if(i < count)                                                       \n" \
"   for (int l = 0; l < precision; l++)                                 \n" \
"   {                                                                   \n" \
"       float z1_real = z_real * z_real - z_imag * z_imag; float z1_imag = 2 * z_real * z_imag; \n" \
"       z_real = z1_real + c_real; z_imag = z1_imag + c_imag; iterations++;                                 \n" \
"       if (z_real * z_real + z_imag * z_imag > 4) { break; }                                \n" \
"   }                                                                   \n" \
"   output[i] = iterations;                                             \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

int main()
{
    int err;                            // error code returned from api calls
    
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned
    
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    //fill dataset with numbers 0 through i
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = i;
    
    // Connect to a compute device
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernel
    kernel = clCreateKernel(program, "Calculate_Julia_Set", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    // Create the input and output arrays in device memory for the calculation
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    sf::RenderWindow window(sf::VideoMode(width, height), "Julia Set Plotter");
    window.setFramerateLimit(30);
    sf::VertexArray pointmap(sf::Points, width * height);
    
    //set up viewing parameters
    float zoom = 275.0f;
    int precision = 300;
    int x_shift = width * 2.5;
    int y_shift = height * 1.2;
    
    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    for (int i = 0; i < width*height; i++)
    {
        pointmap[i].color = sf::Color::Black;
    }
    
    // Set the arguments to the compute kernel
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    int mouse_x = sf::Mouse::getPosition().x;
    int mouse_y = sf::Mouse::getPosition().y;
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &x_shift);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &y_shift);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &precision);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &zoom);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &mouse_x);
    err |= clSetKernelArg(kernel, 8, sizeof(float), &mouse_y);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);
    
    // Read back the results from the device
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    generate_julia_set(pointmap, results, precision);
    std::cout << "Precision: " << precision << std::endl;
    
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        mouse_x = sf::Mouse::getPosition().x;
        mouse_y = sf::Mouse::getPosition().y;
        
        //clear the pointmap
        for (int i = 0; i < width*height; i++)
        {
            pointmap[i].color = sf::Color::Black;
        }
        
        // Set the arguments to the compute kernel
        err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
        err = 0;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &x_shift);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &y_shift);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &precision);
        err |= clSetKernelArg(kernel, 6, sizeof(float), &zoom);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &mouse_x);
        err |= clSetKernelArg(kernel, 8, sizeof(float), &mouse_y);
        
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }
        
        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        global = count;
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err)
        {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }
        
        // Wait for the command commands to get serviced before reading back results
        clFinish(commands);
        
        // Read back the results from the device
        err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        
        generate_julia_set(pointmap, results, precision);
        
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
        {
            std::cout << "manually enter precision, or enter -1 to keep current precision: " << std::endl;
            int new_precision = 0;
            std::cin >> new_precision;
            if (new_precision != -1)
                precision = new_precision;
            for (int i = 0; i < width*height; i++)
            {
                pointmap[i].color = sf::Color::Black;
            }
            
            // Set the arguments to the compute kernel
            err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to write to source array!\n");
                exit(1);
            }
            err = 0;
            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
            err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &x_shift);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &y_shift);
            err |= clSetKernelArg(kernel, 5, sizeof(int), &precision);
            err |= clSetKernelArg(kernel, 6, sizeof(float), &zoom);
            err |= clSetKernelArg(kernel, 7, sizeof(int), &mouse_x);
            err |= clSetKernelArg(kernel, 8, sizeof(float), &mouse_y);
            
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to set kernel arguments! %d\n", err);
                exit(1);
            }
            
            // Execute the kernel over the entire range of our 1d input data set
            // using the maximum number of work group items for this device
            global = count;
            err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            if (err)
            {
                printf("Error: Failed to execute kernel!\n");
                return EXIT_FAILURE;
            }
            
            // Wait for the command commands to get serviced before reading back results
            clFinish(commands);
            
            // Read back the results from the device
            err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array! %d\n", err);
                exit(1);
            }
            
            generate_julia_set(pointmap, results, precision);
            std::cout << "Precision: " << precision << std::endl;
        }
        
        window.clear();
        window.draw(pointmap);
        window.display();
    }
    
    return 0;
}