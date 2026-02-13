// C++ implementation for mandelbrot set fractals
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#define MAXCOUNT 29

char* charset = ".,'~=+:;[/<&?oxOX#abcdefghijk ";

float time_diff(struct timespec *start, struct timespec *end){
    return (end->tv_sec - start->tv_sec) + 1e-9 * (end->tv_nsec - start->tv_nsec);
}

int main(int argc, char** argv)
{
    // Init parameters
    float left, top, right, bottom;

    left = -2.0;
    top = -1.5;
    right = 1.0;
    bottom = 1.5;

    int isVerbose = 0;
    int ySize = 500;
    char* outputFile = "output.txt";
    int opt;
    while((opt = getopt(argc, argv, "y:o:v")) != -1)
    {
        switch(opt)
        {
            case 'y':
                ySize = atoi(optarg);
                break;
            case 'o':
                outputFile = optarg;
                break;
            case 'v':
                isVerbose = 1;
        }
    }

    // OpenMPI initialization
    MPI_Init(NULL, NULL);       // initialize MPI environment
    int worldSize;             // number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int worldRank;             // the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    char processorName[MPI_MAX_PROCESSOR_NAME]; // gets the name of the processor
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);

    // Fractal calculation starts here
    float zx, zy, cx, tempx, cy, epsillonX, epsillonY;
    int x, y;
    int maxX, maxY, count;

    epsillonX = 0.0001;
    epsillonY = (bottom - top) / (float) (worldSize * ySize);
    maxX = (right - left) / epsillonX;
    maxY = (bottom - top) / epsillonY;
    if (isVerbose && worldRank == 0)
    {
        printf("ySize: %d, worldSize: %d\n", ySize, worldSize);
        printf("epsillonY: %f, maxX: %d, maxY: %d\n", epsillonY, maxX, maxY);
    }

    // Split workload based on rank
    int stepYRank = maxY / worldSize;
    if (isVerbose)
    {
        printf("machine name: %s, rank: %d, stepYRank: %d\n", processorName, worldRank, stepYRank);
    }

    // Create array to contain results
    // Array is flattened to 1-d for easier data transfer
    int* results = (int*) calloc(stepYRank * maxX, sizeof(int));

    // Start timer
    struct timespec start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_REALTIME, &start);

    // Scanning every point in that rectangular area.
    // Each point represents a Complex number (x + yi).
    // Iterate that complex number
    for (int i = 1; i <= stepYRank; i++)
    {
        y = (i - 1) * worldSize + worldRank + 1;
        for (x = 1; x <= maxX; x++)
        {
            // c_real
            cx = x * epsillonX + left;

            // c_imaginary
            cy = y * epsillonY + top;

            // z_real
            zx = 0;

            // z_imaginary
            zy = 0;
            count = 0;

            // Calculate whether c(c_real + c_imaginary) belongs
            // to the Mandelbrot set or not and draw a pixel
            // at coordinates (x, y) accordingly
            // If you reach the Maximum number of iterations
            // and If the distance from the origin is
            // greater than 2 exit the loop
            while ((zx * zx + zy * zy < 4) && (count < MAXCOUNT))
            {
                // Calculate Mandelbrot function
                // z = z*z + c where z is a complex number

                // tempx = z_real*_real - z_imaginary*z_imaginary + c_real
                tempx = zx * zx - zy * zy + cx;

                // 2*z_real*z_imaginary + c_imaginary
                zy = 2 * zx * zy + cy;

                // Updating z_real = tempx
                zx = tempx;

                // Increment count
                count = count + 1;
            }

            // Save results on 1-d array using mapping
            results[(i - 1) * maxX + (x - 1)] = count;
        }
    }

    // Gather results
    int* allResults;
    if (worldRank == 0)
    {
        allResults = (int*) calloc(maxY * maxX, sizeof(int));
    }
    MPI_Gather(results, stepYRank * maxX, MPI_INT, allResults, stepYRank * maxX, MPI_INT, 0, MPI_COMM_WORLD);
    clock_gettime(CLOCK_REALTIME, &end);

    if (worldRank == 0)
    {
        // Write results and time
        printf("Total time elapsed: %0.8f sec\n",
            time_diff(&start, &end));
        printf("Writing mandelbrot output to %s ...\n", outputFile);
        FILE *fptr = fopen(outputFile, "w");
        for (int i = 0; i < maxY; i++)
        {
            for (int j = 0; j < maxX; j++)
            {
                fprintf(fptr, "%c", charset[allResults[i * maxX + j]]);
            }
            fprintf(fptr, "\n");
        }
        printf("DONE!\n");
    }

    MPI_Finalize(); // finish MPI environment

    return 0;
}