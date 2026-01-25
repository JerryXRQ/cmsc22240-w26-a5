#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

#define MAX_ITERS 1000
#define EPS 1e-6

// Complex plane window
#define XMIN -2.0
#define XMAX  2.0
#define YMIN -2.0
#define YMAX  2.0

// Resolution in X direction
#define EPS_X 0.002



// Function to determine closest root of z^3 - 1
int closest_root(double complex z) {
    double complex roots[3] = {1.0 + 0.0*I,
                               -0.5 + 0.86602540378*I,
                               -0.5 - 0.86602540378*I};
    int min_idx = 0;
    double min_dist = cabs(z - roots[0]);
    for (int i = 1; i < 3; i++) {
        double dist = cabs(z - roots[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }
    return min_idx;
}

// Time difference helper
double time_diff(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + 1e-9*(end->tv_nsec - start->tv_nsec);
}

int main(int argc, char** argv)
{
    int ySize = 500;          // configurable
    char* outputFile = "newton.ppm";
    int isVerbose = 0;

    int opt;
    while((opt = getopt(argc, argv, "y:o:v")) != -1) {
        switch(opt) {
            case 'y': ySize = atoi(optarg); break;
            case 'o': outputFile = optarg; break;
            case 'v': isVerbose = 1; break;
        }
    }


    // MPI initialization
    MPI_Init(NULL, NULL);
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);
    if (isVerbose)
        printf("Machine: %s, Rank: %d of %d\n", processorName, worldRank, worldSize);

    // Calculate steps and max dimensions
    double epsY = (YMAX - YMIN)/(ySize * worldSize);
    int maxX = (int)((XMAX - XMIN) / EPS_X);
    int maxY = (int)((YMAX - YMIN)/epsY);

    int maxYRank = maxY / worldSize;
    int startYRank = worldRank * maxYRank;

    if (isVerbose)
        printf("Rank %d handles rows %d to %d\n", worldRank, startYRank, startYRank + maxYRank);

    // Allocate local image buffer (RGB)
    unsigned char* local_img = (unsigned char*) calloc(maxYRank * maxX * 3, sizeof(unsigned char));

    struct timespec startTime, endTime;
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_REALTIME, &startTime);

    // Main Newton iteration
    for (int y = startYRank; y < startYRank + maxYRank; y++) {
        double im = YMIN + y * (YMAX - YMIN) / maxY;
        for (int x = 0; x < maxX; x++) {
            double re = XMIN + x * EPS_X;
            double complex z = re + im * I;

            int iter;
            for (iter = 0; iter < MAX_ITERS; iter++) {
                if (cabs(z*z*z - 1) < EPS) break;
                z = z - (z*z*z - 1)/(3.0*z*z);
            }

            int root = closest_root(z);
            int idx = (y - startYRank) * maxX * 3 + x * 3;

            // Color by root + iteration count (brightness)
            unsigned char brightness = (unsigned char)(255 - iter * 255 / MAX_ITERS);
            if (root == 0) {        // Red root
                local_img[idx+0] = brightness;
                local_img[idx+1] = 0;
                local_img[idx+2] = 0;
            } else if (root == 1) { // Green root
                local_img[idx+0] = 0;
                local_img[idx+1] = brightness;
                local_img[idx+2] = 0;
            } else {                 // Blue root
                local_img[idx+0] = 0;
                local_img[idx+1] = 0;
                local_img[idx+2] = brightness;
            }
        }
    }

    // Gather results
    unsigned char* full_img = NULL;
    if (worldRank == 0)
        full_img = (unsigned char*) calloc(maxY * maxX * 3, sizeof(unsigned char));

    MPI_Gather(local_img, maxYRank * maxX * 3, MPI_UNSIGNED_CHAR,
               full_img, maxYRank * maxX * 3, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    clock_gettime(CLOCK_REALTIME, &endTime);

    if (worldRank == 0) {
        printf("Total time elapsed: %0.6f sec\n", time_diff(&startTime, &endTime));
        printf("Writing Newton fractal to %s ...\n", outputFile);

        FILE* fptr = fopen(outputFile, "wb");
        fprintf(fptr, "P6\n%d %d\n255\n", maxX, maxY);
        fwrite(full_img, 3, maxX*maxY, fptr);
        fclose(fptr);
        printf("DONE!\n");
        free(full_img);
    }

    free(local_img);
    MPI_Finalize();
    return 0;
}
