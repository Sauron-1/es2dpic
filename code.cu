#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <time.h>


/*************************
 * Simulation parameters *
 *************************/
#define SAVE_PATH "data"
#define USE_NETCDF 0

#define N_STEPS 100
#define N_SAVE 99

#define DT 0.02

#define NX 256
#define NY 256

#define NPARTX 5
#define NPARTY 5

#define N_SPECIES 3

#define AX (13./15)
#define AY (13./15)

const double ms[N_SPECIES] = {1*0.98, 1*0.02, 1836};
const double qs[N_SPECIES] = {-1*0.98, -1*0.02, 1};
const double vths[N_SPECIES][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {0.0233380014, 0.0233380014, 0.0233380014}};
const double vdrs[N_SPECIES][3] = {
    {0, 0, 0},
    {10, 0, 0},
    {0, 0, 0}};

const double B[3] = {1, 0, 0};

#define NPART (NX*NY*NPARTX*NPARTY)

#define PI 3.1415926535897932384626433832795
#define EPSILON0 (NPARTX*NPARTY)

/*****************************
 * End simulation parameters *
 *****************************/

#if(USE_NETCDF == 1)
#   include <netcdf.h>
#endif // USE_NETCDF == 1


struct FieldSolverHandle {

    cufftDoubleComplex *NeFFT, *fieldEFFT, *fieldPhiFFT;

    cufftHandle planD2Z, planZ2D;

    double *filter;

    double *kxs, *kys;

};


__global__
void calNumDensity(
        const double *pos, const int n_part,
        double *den, const int nx, const int ny) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < n_part; idx += stride) {
        double dx = pos[idx*2];
        double dy = pos[idx*2 + 1];

        const int x = (int)(dx+0.5);
        const int y = (int)(dy+0.5);

        double valx[3], valy[3];

        dx -= x;
        dy -= y;

        valx[0] = (0.5-dx) * (0.5-dx) * 0.5;
        valx[1] = 0.75 - dx*dx;
        valx[2] = (0.5+dx) * (0.5+dx) * 0.5;

        valy[0] = (0.5-dy) * (0.5-dy) * 0.5;
        valy[1] = 0.75 - dy*dy;
        valy[2] = (0.5+dy) * (0.5+dy) * 0.5;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int gx = (x+i-1) % nx;
                int gy = (y+j-1) % ny;
                gx = gx >= 0 ? gx : nx + gx;
                gy = gy >= 0 ? gy : ny + gy;
                atomicAdd(den+gx*ny+gy, valx[i]*valy[j]);
            }
        }

    }
}


__global__
void moveParticles(
        double m, double q, double dt,
        double *pos, double *vel, const int n_part,
        const double *E, const double *B0,
        const int nx , const int ny) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const double qtm = q*dt / m / 2;
    const double Bs[3] = {B0[0]*qtm, B0[1]*qtm, B0[2]*qtm};

    for (int idx = tid; idx < n_part; idx += stride) {
        double dx = pos[idx*2];
        double dy = pos[idx*2 + 1];

        int x = (int)(dx+0.5);
        int y = (int)(dy+0.5);

        double valx[3], valy[3];

        dx -= x;
        dy -= y;

        valx[0] = (0.5-dx) * (0.5-dx) * 0.5;
        valx[1] = 0.75 - dx*dx;
        valx[2] = (0.5+dx) * (0.5+dx) * 0.5;

        valy[0] = (0.5-dy) * (0.5-dy) * 0.5;
        valy[1] = 0.75 - dy*dy;
        valy[2] = (0.5+dy) * (0.5+dy) * 0.5;

        double F[2] = {0, 0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int gx = (x+i-1) % nx;
                int gy = (y+j-1) % ny;
                gx = gx >= 0 ? gx : nx + gx;
                gy = gy >= 0 ? gy : ny + gy;
                for (int k = 0; k < 2; k++) {
                    F[k] += E[k*nx*ny + gx*ny + gy] * valx[i] * valy[j];
                }
            }
        }

        F[0] *= qtm;
        F[1] *= qtm;

        vel[idx*3+0] += F[0];
        vel[idx*3+1] += F[1];

        double vel0[3];
        double f = 2. / (1+Bs[0]*Bs[0]+Bs[1]*Bs[1]+Bs[2]*Bs[2]);

        vel0[0] = f*(vel[3*idx+0] + vel[3*idx+1]*Bs[2] - vel[3*idx+2]*Bs[1]);
        vel0[1] = f*(vel[3*idx+1] + vel[3*idx+2]*Bs[0] - vel[3*idx+0]*Bs[2]);
        vel0[2] = f*(vel[3*idx+2] + vel[3*idx+0]*Bs[1] - vel[3*idx+1]*Bs[0]);

        vel[3*idx+0] += vel0[1]*Bs[2] - vel0[2]*Bs[1] + F[0];
        vel[3*idx+1] += vel0[2]*Bs[0] - vel0[0]*Bs[2] + F[1];
        vel[3*idx+2] += vel0[0]*Bs[1] - vel0[1]*Bs[0];

        pos[2*idx+0] += vel[3*idx+0] * dt;
        pos[2*idx+1] += vel[3*idx+1] * dt;

        if (pos[2*idx+0] < -0.5 || pos[2*idx+0] >= nx - 0.5) {
            pos[2*idx+0] = fmod(pos[2*idx+0]+0.5, (double)nx);
            pos[2*idx+0] = pos[2*idx+0] >= 0 ?
                pos[2*idx+0] - 0.5 : nx + pos[2*idx+0] - 0.5;
        }
        if (pos[2*idx+1] < -0.5 || pos[2*idx+1] >= ny - 0.5) {
            pos[2*idx+1] = fmod(pos[2*idx+1]+0.5, (double)ny);
            pos[2*idx+1] = pos[2*idx+1] >= 0 ?
                pos[2*idx+1] - 0.5 : ny + pos[2*idx+1] - 0.5;
        }
    }


}


/*******************************************************
 * Functions to solve electron field, using FFT method *
 *******************************************************/
__global__
void solvePossionFFT(
        const cufftDoubleComplex *NeFFT,
        cufftDoubleComplex *fieldEFFT,
        cufftDoubleComplex *fieldPhiFFT,
        const double *filter, const double *kxs, const double *kys,
        int nx, int ny) {

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int ny_fft = ny/2 + 1;

    for (int idx = tid; idx < nx*ny_fft; idx += stride) {
        const int x = idx / ny_fft;
        const int y = idx % ny_fft;

        const double kx = kxs[x], ky = kys[y];

        const double diff2 = -(kx*kx + ky*ky)*EPSILON0*nx*ny;

        if (fabs(diff2) < 1e-50) {
            fieldPhiFFT[idx] = make_cuDoubleComplex(0, 0);
        }
        else {
            fieldPhiFFT[idx] = cuCmul(NeFFT[idx],
                    make_cuDoubleComplex(filter[idx]/diff2, 0));
        }
        fieldEFFT[idx] = cuCmul(fieldPhiFFT[idx],
                make_cuDoubleComplex(0, kx));
        fieldEFFT[idx+nx*ny_fft] = cuCmul(fieldPhiFFT[idx],
                make_cuDoubleComplex(0, ky));
    }

}


__host__
void solveField(
        FieldSolverHandle *handle,
        double *fieldE, double *fieldPhi, double *Ne,
        int nx, int ny) {

    const int ny_fft = ny/2 + 1;

    cufftExecD2Z(handle->planD2Z, Ne, handle->NeFFT);

    solvePossionFFT<<<24, 1024>>>(
            handle->NeFFT, handle->fieldEFFT, handle->fieldPhiFFT,
            handle->filter, handle->kxs, handle->kys, nx, ny);

    cufftExecZ2D(handle->planZ2D,
            handle->fieldPhiFFT, fieldPhi);
    cufftExecZ2D(handle->planZ2D,
            handle->fieldEFFT, fieldE);
    cufftExecZ2D(handle->planZ2D,
            handle->fieldEFFT+nx*ny_fft, fieldE+nx*ny);

}


__global__
void initSolverHelper(
        double *filter, double *kxs, double *kys, int nx, int ny) {

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int ny_fft = ny/2 + 1;

    for (int idx = tid; idx < nx*ny_fft; idx += stride) {
        const int x = idx / ny_fft;
        const int y = idx % ny_fft;

        double kx = 2.*x*PI / nx;
        double ky = 2.*y*PI / ny;

        if (2*x > nx) {
            kx = 2*(x - nx)*PI / nx;
        }
        if (2*y > ny) {
            ky = 2*(y - ny)*PI / ny;
        }

        filter[idx] = exp(-(AX*AX*kx*kx + AY*AY*ky*ky));
        if (y == 0) {
            kxs[x] = kx;
        }
        if (x == 0) {
            kys[y] = ky;
        }
    }

}


__host__
int initFieldSolverHandle(FieldSolverHandle *handle, int nx, int ny) {
    const int ny_fft = ny/2 + 1;
    int flag = 0;
    flag |= cudaMalloc(&handle->NeFFT, nx*ny_fft*sizeof(cufftDoubleComplex));
    flag |= cudaMalloc(&handle->fieldPhiFFT, nx*ny_fft*sizeof(cufftDoubleComplex));
    flag |= cudaMalloc(&handle->fieldEFFT, 2*nx*ny_fft*sizeof(cufftDoubleComplex));

    flag |= cudaMalloc(&handle->filter, nx*ny_fft*sizeof(double));
    flag |= cudaMalloc(&handle->kxs, nx*sizeof(double));
    flag |= cudaMalloc(&handle->kys, ny_fft*sizeof(double));

    if (flag == 0) {
        initSolverHelper<<<24, 1024>>>(
                handle->filter, handle->kxs, handle->kys, nx, ny);
    }

    cufftPlan2d(&handle->planD2Z, nx, ny, CUFFT_D2Z);
    cufftPlan2d(&handle->planZ2D, nx, ny, CUFFT_Z2D);
    return flag;
}



/******************************
 * Functions for initializing *
 ******************************/
__global__
void uniformInit(double *pos, int n_part) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < n_part; idx += stride) {
        pos[idx*2+0] = (double)(idx/(NPARTY*NY)) / NPARTX - 0.5;
        pos[idx*2+1] = (double)(idx%(NPARTY*NY)) / NPARTY - 0.5;
    }

}



__global__
void setVelocity(double *vel, int n_part, const double *vth, const double *vdr) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < n_part; idx += stride) {
        vel[3*idx+0] = vel[3*idx+0] * vth[0] + vdr[0];
        vel[3*idx+1] = vel[3*idx+1] * vth[1] + vdr[1];
        vel[3*idx+2] = vel[3*idx+2] * vth[2] + vdr[2];
    }

}


__global__
void zero(double *arr, const int n) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < n; idx += stride) {
        arr[idx] = 0;
    }
}



/**************************************
 * Struct and functions to save data. *
 **************************************/
struct SaveHandle {
    double *Ne;
    double *E;
    char filename[100];
};


void initSaveHandle(SaveHandle *handle, int nx, int ny) {
    handle->Ne = (double *)malloc(nx*ny*sizeof(double));
    handle->E = (double *)malloc(2*nx*ny*sizeof(double));
}

#if(USE_NETCDF == 1)
int saveStatus(SaveHandle *handle, double *Ne, double *E,
        int nx, int ny, int step) {
    int ncid, ne_id, Ex_id, Ey_id, ok=0, dimid[2];
    sprintf(handle->filename, "%s/data%05d.nc", SAVE_PATH, step);
    ok |= nc_create(handle->filename, NC_CLOBBER, &ncid);

    cudaMemcpy(handle->Ne, Ne,
            sizeof(double)*nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(handle->E, E,
            2*sizeof(double)*nx*ny, cudaMemcpyDeviceToHost);
    
    ok |= nc_def_dim(ncid, "0", nx, dimid+0);
    ok |= nc_def_dim(ncid, "1", ny, dimid+1);

    ok |= nc_def_var(ncid, "Ne", NC_DOUBLE, 2, dimid, &ne_id);
    ok |= nc_def_var(ncid, "Ex", NC_DOUBLE, 2, dimid, &Ex_id);
    ok |= nc_def_var(ncid, "Ey", NC_DOUBLE, 2, dimid, &Ey_id);

    ok |= nc_enddef(ncid);

    ok |= nc_put_var_double(ncid, ne_id, handle->Ne);
    ok |= nc_put_var_double(ncid, Ex_id, handle->E);
    ok |= nc_put_var_double(ncid, Ey_id, handle->E+nx*ny);

    ok |= nc_close(ncid);

    return ok;
}
#else // USE_NETCDF == 1
int saveStatus(SaveHandle *handle, double *Ne, double *E,
        int nx, int ny, int step) {

    sprintf(handle->filename, "%s/data%05d.csv", SAVE_PATH, step);

    FILE *fp = fopen(handle->filename, "w");
    if (fp == NULL) {
        return 1;
    }

    fprintf(fp, "x,y,Ne,Ex,Ey\n");

    cudaMemcpy(handle->Ne, Ne,
            sizeof(double)*nx*ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(handle->E, E,
            2*sizeof(double)*nx*ny, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int idx = i*ny + j;
            fprintf(fp, "%d,%d,%lf,%lf,%lf\n", i, j, 
                    handle->Ne[idx], 
                    handle->E[idx], 
                    handle->E[idx+nx*ny]);
        }
    }
    fclose(fp);

    return 0;
}
#endif // USE_NETCDF == 1


int main() {
    
    double *pos[N_SPECIES], *vel[N_SPECIES], *den[N_SPECIES];
    double *Ne, *fieldE, *fieldPhi;

    double *vths_dev[N_SPECIES], *vdrs_dev[N_SPECIES], *B_dev;

    double Ek[N_SPECIES], EE, Ek_sum;

    int flag = 0;

    // Allocate device memory
    for (int s = 0; s < N_SPECIES; s++) {
        flag |= cudaMalloc(vths_dev+s, 3*sizeof(double));
        flag |= cudaMalloc(vdrs_dev+s, 3*sizeof(double));
        cudaMemcpy(vths_dev[s], vths[s],
                3*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(vdrs_dev[s], vdrs[s],
                3*sizeof(double), cudaMemcpyHostToDevice);
    }
    flag |= cudaMalloc(&B_dev, 3*sizeof(double));
    cudaMemcpy(B_dev, B,
            3*sizeof(double), cudaMemcpyHostToDevice);

    for (int s = 0; s < N_SPECIES; s++) {
        flag |= cudaMalloc(&pos[s], 2*NPART*sizeof(double));
        flag |= cudaMalloc(&vel[s], 3*NPART*sizeof(double));
        flag |= cudaMalloc(&den[s], NX*NY*sizeof(double));
    }

    flag |= cudaMalloc(&Ne, NX*NY*sizeof(double));
    flag |= cudaMalloc(&fieldE, 2*NX*NY*sizeof(double));
    flag |= cudaMalloc(&fieldPhi, NX*NY*sizeof(double));

    FieldSolverHandle fieldHandle;
    flag |= initFieldSolverHandle(&fieldHandle, NX, NY);

    SaveHandle saveHandle;
    initSaveHandle(&saveHandle, NX, NY);

    if (flag != 0) {
        printf("ERROR: allocate failed\n");
        exit(1);
    }

    // Init cuBLAS and cuRAND handles
    cublasHandle_t blasHandle;
    cublasCreate(&blasHandle);

    curandGenerator_t curandHandle;
    curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandHandle, time(NULL));

    // Init postition and velocity
    for (int s = 0; s < N_SPECIES; s++) {
        uniformInit<<<24, 768>>>(pos[s], NPART);
    }
    for (int s = 0; s < N_SPECIES; s++) {
        curandGenerateNormalDouble(curandHandle, vel[s], 3*NPART, 0, 1);
        setVelocity<<<24, 768>>>(vel[s], NPART, vths_dev[s], vdrs_dev[s]);
    }
    cudaDeviceSynchronize();
    FILE *output = fopen("energy.dat", "w");

    // Calculate grid/block size
    int grd_mv, blk_mv;
    cudaOccupancyMaxPotentialBlockSize(&grd_mv, &blk_mv, moveParticles, 0, 0);
    printf("Will use <<<%d, %d>>> to configure function moveParticles\n", grd_mv, blk_mv);
    int grd_den, blk_den;
    cudaOccupancyMaxPotentialBlockSize(&grd_den, &blk_den, calNumDensity, 0, 0);
    printf("Will use <<<%d, %d>>> to configure function calNumDensity\n", grd_den, blk_den);

    clock_t start, end;

    start = clock();
    const int nstep = N_STEPS - N_STEPS % N_SAVE;
    printf("Step \t\t");
    for (int i = 0; i < N_STEPS; i++) {
        printf("Ek %d\t\t", i);
    }
    printf("Ek \t\t EE \t\t E\n");
    for (int step = 0; step <= nstep; step++) {
        
        printf("%d\t\t", step);
        fprintf(output, "%d\t\t", step);

        for (int s = 0; s < N_SPECIES; s++) {
            zero<<<24, 1024>>>(den[s], NX*NY);
        }
        zero<<<24, 1024>>>(Ne, NX*NY);
        cudaDeviceSynchronize();
        for (int s = 0; s < N_SPECIES; s++) {
            calNumDensity<<<grd_den, blk_den>>>(pos[s], NPART, den[s], NX, NY);
        }
        cudaDeviceSynchronize();

        for (int s = 0; s < N_SPECIES; s++) {
            cublasDaxpy(blasHandle, NX*NY,
                    qs+s, den[s], 1, Ne, 1);
        }

        solveField(&fieldHandle, fieldE, fieldPhi, Ne, NX, NY);
        cudaDeviceSynchronize();

        for (int s = 0; s < N_SPECIES; s++) {
            moveParticles<<<grd_mv, blk_mv>>>(ms[s], qs[s], DT,
                    pos[s], vel[s], NPART,
                    fieldE, B_dev, NX, NY);
        }
        cudaDeviceSynchronize();

        // Save charge density and electric field
        if (step % N_SAVE == 0) {
            saveStatus(&saveHandle, Ne, fieldE, NX, NY, step/ N_SAVE);
        }

        // Print diagnosis info
        Ek_sum = 0;
        for (int s = 0; s < N_SPECIES; s++) {
            Ek[s] = 0;
            cublasDdot(blasHandle, NPART*3,
                    vel[s], 1, vel[s], 1, Ek+s);
            Ek[s] *= 0.5 * ms[s] / NPART;
            Ek_sum += Ek[s];
        }
        
        cublasDdot(blasHandle, NX*NY,
                Ne, 1, fieldPhi, 1, &EE);
        EE /= -NPART*2;

        for (int s = 0; s < N_SPECIES; s++) {
            printf("%e\t", Ek[s]);
            fprintf(output, "%e\t", Ek[s]);
        }
        printf("%e\t%e\t%e\n", Ek_sum, EE, Ek_sum+EE);
        fprintf(output, "%e\t%e\t%e\n", Ek_sum, EE, Ek_sum+EE);

    }
    end = clock();
    printf("Finished main loop in %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    fclose(output);

}
