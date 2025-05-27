#include <stdio.h>
#include <complex.h>
#include <omp.h>

void SOCcompute_ss(const double complex* zeff, 
    const double complex* civecs,
    double complex* b_data, 
    const int* alphadetcoupl, 
    const int* betadetcoupl, 
    const int* shapearr) {

    // Reading the shapearr to get the shapes of different arrays
    // b_data is the output array, 
    // alphadetcoupl and betadetcoupl are the coupling arrays
    int b0shp = shapearr[0];
    int b1shp = shapearr[1]; 
    int b2shp = shapearr[2]; 
    int b3shp = shapearr[3];
    int a0shp = shapearr[4]; 
    int a1shp = shapearr[5];
    int nao = shapearr[6];
    int c1shp = shapearr[7];
    int c2shp = shapearr[8];
    int bsize = b0shp * b1shp * b2shp * b3shp;
    int b1b2b3shp = b1shp * b2shp * b3shp;
    int b2b3shp = b2shp * b3shp;

    // Initializing the b_data
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bsize; ++i) {
        b_data[i] = 0.0 + 0.0 * I;
    }

    #pragma omp parallel for collapse(4)
    for (int b0 = 0; b0 < b0shp; ++b0) {
        for (int b1 = 0; b1 < b1shp; ++b1) {
            for (int b2 = 0; b2 < b2shp; ++b2) {
                for (int b3 = 0; b3 < b3shp; ++b3) {
                    double complex sum = 0.0 + 0.0 * I;

                    for (int aa = 0; aa < a0shp; ++aa) {
                        int a0_idx = b2 * a0shp * 4 + aa * 4;
                        int z_idx = b0 * nao * nao + alphadetcoupl[a0_idx] * nao + alphadetcoupl[a0_idx + 1];
                        int c_idx = b1 * (c1shp * c2shp) + alphadetcoupl[a0_idx + 2] * c2shp + b3;
                        sum += zeff[z_idx] * civecs[c_idx] * (double)alphadetcoupl[a0_idx + 3];
                    }

                    for (int ab = 0; ab < a1shp; ++ab) {
                        int a1_idx = b3 * a1shp * 4 + ab * 4;
                        int z_idx = b0 * nao * nao + betadetcoupl[a1_idx] * nao + betadetcoupl[a1_idx + 1];
                        int c_idx = b1 * (c1shp * c2shp) + b2 * c2shp + betadetcoupl[a1_idx + 2];
                        sum -= zeff[z_idx] * civecs[c_idx] * (double)betadetcoupl[a1_idx + 3];
                    }

                    // Store the result to b_data
                    int b_idx = b0 * b1b2b3shp + b1 * b2b3shp + b2 * b3shp + b3;
                    b_data[b_idx] = sum;
                }
            }
        }
    }
}

void SOCcompute_ssp(const double complex* zeff,
    const double complex* civecs,
    double complex* b_data,
    const int* alphadetcoupl,
    const int* betadetcoupl,
    const int* shapearr) {

    int b0shp = shapearr[0];
    int b1shp = shapearr[1]; 
    int b2shp = shapearr[2]; 
    int b3shp = shapearr[3];
    int a0shp = shapearr[4]; 
    int a1shp = shapearr[5];
    int nao = shapearr[6];
    int c1shp = shapearr[7];
    int c2shp = shapearr[8];
    int bsize = b0shp * b1shp * b2shp * b3shp;
    int b1b2b3shp = b1shp * b2shp * b3shp;
    int b2b3shp = b2shp * b3shp;

    // Initializing the b_data
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < bsize; ++i) {
        b_data[i] = 0.0 + 0.0 * I;
    }

    #pragma omp parallel for collapse(4)
    for (int b0 = 0; b0 < b0shp; ++b0) {
        for (int b1 = 0; b1 < b1shp; ++b1) {
            for (int b2 = 0; b2 < b2shp; ++b2) {
                for (int b3 = 0; b3 < b3shp; ++b3) {
                    double complex sum = 0.0 + 0.0 * I;

                    for (int aa = 0; aa < a0shp; ++aa) {
                        int a0_idx = b2 * a0shp * 4 + aa * 4;
                        for (int ab = 0; ab < a1shp; ++ab) {
                            int a1_idx = b3 * a1shp * 4 + ab * 4;
                            int z_idx = b0 * nao * nao + betadetcoupl[a1_idx] * nao + alphadetcoupl[a0_idx + 1];
                            int c_idx = b1 * (c1shp * c2shp) + alphadetcoupl[a0_idx + 2] * c2shp + betadetcoupl[a1_idx + 2];
                            sum += civecs[c_idx] * zeff[z_idx] * (double)alphadetcoupl[a0_idx + 3] * (double)betadetcoupl[a1_idx + 3];
                        }
                    }
                    // Store the result to b_data
                    int b_idx = b0 * b1b2b3shp + b1 * b2b3shp + b2 * b3shp + b3;
                    b_data[b_idx] = sum;
                }
            }
        }
    }
}

void SOCcompute_ssm(const double complex* zeff,
    const double complex* civecs,
    double complex* b_data,
    const int* alphadetcoupl,
    const int* betadetcoupl,
    const int* shapearr) {
    
    int b0shp = shapearr[0];
    int b1shp = shapearr[1]; 
    int b2shp = shapearr[2]; 
    int b3shp = shapearr[3];
    int a0shp = shapearr[4]; 
    int a1shp = shapearr[5];
    int nao = shapearr[6];
    int c1shp = shapearr[7];
    int c2shp = shapearr[8];
    int bsize = b0shp * b1shp * b2shp * b3shp;
    int b1b2b3shp = b1shp * b2shp * b3shp;
    int b2b3shp = b2shp * b3shp;

    // Initialize b_data to zero
    #pragma omp parallel for
    for (int i = 0; i < bsize; ++i) {
        b_data[i] = 0.0 + 0.0 * I;
    }

    #pragma omp parallel for collapse(4)
    for (int b0 = 0; b0 < b0shp; ++b0) {
        for (int b1 = 0; b1 < b1shp; ++b1) {
            for (int b2 = 0; b2 < b2shp; ++b2) {
                for (int b3 = 0; b3 < b3shp; ++b3) {
                    double complex sum = 0.0 + 0.0 * I;
                    
                    for (int aa = 0; aa < a0shp; ++aa) {
                        int a0_idx = b2 * a0shp * 4 + aa * 4;
                        for (int ab = 0; ab < a1shp; ++ab) {
                            int a1_idx = b3 * a1shp * 4 + ab * 4;
                            int z_idx = b0 * nao * nao + alphadetcoupl[a0_idx] * nao + betadetcoupl[a1_idx + 1];
                            int c_idx = b1 * (c1shp * c2shp) + alphadetcoupl[a0_idx + 2] * c2shp + betadetcoupl[a1_idx + 2];
                            sum += civecs[c_idx] * zeff[z_idx] * (double)alphadetcoupl[a0_idx + 3] * (double)betadetcoupl[a1_idx + 3];
                        }
                    }
                    // Store the result to b_data
                    int b_idx = b0 * b1b2b3shp + b1 * b2b3shp + b2 * b3shp + b3;
                    b_data[b_idx] = sum;
                }
            }
        }
    }
}
