/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef KOKKOSBLAS3_COMMON_H_
#define KOKKOSBLAS3_COMMON_H_
#include "KokkosKernels_default_types.hpp"

#include <iostream>
#include <fstream>

#define DEFAULT_TEST BLAS
#define DEFAULT_LOOP SERIAL
#define DEFAULT_MATRIX_START 10
#define DEFAULT_MATRIX_STOP 2430
#define DEFAULT_STEP 3
#define DEFAULT_WARM_UP_N 100
#define DEFAULT_N 100
#define DEFAULT_OUT &std::cout
#define DEFAULT_BLAS_ROUTINES "trmm,"

struct perf_test_trmm_args {
  std::string trmm_args;
  default_scalar alpha;
};
typedef struct perf_test_trmm_args pt_trmm_args_t;

//ADD MORE BLAS ROUTINE ARG STRUCTS HERE.

struct blas_args {
  pt_trmm_args_t trmm;
  //ADD MORE BLAS ROUTINES HERE
};
typedef struct blas_args blas_args_t;

typedef enum BLAS_ROUTINES {
    TRMM,
    //ADD MORE BLAS ROUTINES HERE
    BLAS_ROUTINES_N
} blas_routines_e;

static std::string blas_routines_e_str[BLAS_ROUTINES_N] = {
  "trmm"
  //ADD MORE BLAS ROUTINES HERE
};

typedef enum LOOP {
  SERIAL,
  PARALLEL,
  LOOP_N
} loop_e;

static std::string loop_e_str[LOOP_N] = {
  "SERIAL",
  "PARALLEL"
};

typedef enum TEST {
  BLAS,
  BATCHED,
  TEST_N
} test_e;

static std::string test_e_str[TEST_N] {
  "BLAS",
  "BATCHED"
};

struct matrix_dim {
  int m, n;
};
typedef struct matrix_dim matrix_dim_t;

struct matrix_dims {
  matrix_dim_t a, b, c;
};
typedef struct matrix_dims matrix_dims_t;

struct perf_test_options {
  test_e test;
  loop_e loop;
  matrix_dims_t start;
  matrix_dims_t stop;
  uint32_t step;
  uint32_t warm_up_n;
  uint32_t n;
  std::ostream* out;
  std::string out_file;
  blas_args_t blas_args;
  std::string blas_routines;
};
typedef struct perf_test_options options_t;
#endif // KOKKOSBLAS3_COMMON_H_
