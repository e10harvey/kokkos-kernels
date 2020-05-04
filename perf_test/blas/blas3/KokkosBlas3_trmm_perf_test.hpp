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
#ifndef KOKKOSBLAS_TRMM_PERF_TEST_H_
#define KOKKOSBLAS_TRMM_PERF_TEST_H_

#include <complex.h>

#include "KokkosKernels_default_types.hpp"

#include <KokkosBlas3_trmm.hpp>

/*****************************************************************************/
#define DEFAULT_TEST BLAS
#define DEFAULT_LOOP SERIAL
#define DEFAULT_MATRIX_START 10
#define DEFAULT_MATRIX_STOP 2430
#define DEFAULT_STEP 3
#define DEFAULT_WARM_UP_N 100
#define DEFAULT_N 100
#define DEFAULT_TRMM_ARGS "LUNU"
#define DEFAULT_TRMM_ALPHA 1.0

struct matrix_dim {
  int m, n;
};
typedef struct matrix_dim matrix_dim_t;

typedef enum TEST {
  BLAS,
  BATCHED,
  TEST_N
} test_e;

static std::string test_e_str[TEST_N] {
  "BLAS",
  "BATCHED"
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

struct trmm_perf_test_options {
  test_e test;
  loop_e loop;
  matrix_dim_t start;
  matrix_dim_t stop;
  uint32_t step;
  uint32_t warm_up_n;
  uint32_t n;
  std::string trmm_args;
  default_scalar alpha;
};
typedef struct trmm_perf_test_options options_t;

/*****************************************************************************/
static void __print_trmm_perf_test_options(options_t options)
{
  printf("options.test      = %s\n", test_e_str[options.test].c_str());
  printf("options.loop      = %s\n", loop_e_str[options.loop].c_str());
  printf("options.start     = %dx%d\n", options.start.m, options.start.n);
  printf("options.stop      = %dx%d\n", options.stop.m, options.stop.n);
  printf("options.step      = %d\n", options.step);
  printf("options.warm_up_n = %d\n", options.warm_up_n);
  printf("options.n         = %d\n", options.n);
  printf("options.trmm_args = %s\n", options.trmm_args.c_str());
  if (std::is_same<double, default_scalar>::value)
    printf("options.alpha     = %lf\n", options.alpha);
  else if (std::is_same<float, default_scalar>::value)
    printf("options.alpha     = %f\n", options.alpha);
  //else if (std::is_same<Kokkos::complex<double>, default_scalar>::value)
  //  printf("options.alpha     = %lf+%lfi\n", creal(options.alpha), cimag(options.alpha));
  //else if (std::is_same<Kokkos::complex<float>, default_scalar>::value)
  //  printf("options.alpha     = %lf+%lfi\n", crealf(options.alpha), cimagf(options.alpha));
  std::cout << "SCALAR:" << typeid(default_scalar).name() <<
               ", LAYOUT:" << typeid(default_layout).name() <<
               ", DEVICE:." << typeid(default_device).name() <<
  std::endl;
}

/*****************************************************************************/
void __do_trmm_serial_blas(uint32_t warm_up_n, uint32_t n)
{
  printf("STATUS: %s.\n", __func__);
  return;
}

void __do_trmm_serial_batched(uint32_t warm_up_n, uint32_t n)
{
  printf("STATUS: %s.\n", __func__);
  return;
}

void __do_trmm_parallel_blas(uint32_t warm_up_n, uint32_t n)
{
  printf("STATUS: %s.\n", __func__);
  return;
}

void __do_trmm_parallel_batched(uint32_t warm_up_n, uint32_t n)
{
  printf("STATUS: %s.\n", __func__);
  return;
}

/*****************************************************************************/
void __do_setup(uint32_t n, matrix_dim_t dim)
{
  printf("STATUS: %s.\n", __func__);
  using view_type_a = Kokkos::View<default_scalar**, default_layout, default_device>;
  using view_type_b = Kokkos::View<default_scalar**, default_layout, default_device>;
  
  return;
}

/*****************************************************************************/
void __do_loop_and_invoke(options_t options, 
                          void (*fn)(uint32_t, uint32_t))
{
  matrix_dim_t cur_dim;
  printf("STATUS: %s.\n", __func__);

  for (cur_dim = options.start;
        cur_dim.m <= options.stop.m && cur_dim.n <= options.stop.n;
        cur_dim.m *= options.step, cur_dim.n *= options.step) {
        __do_setup(options.n, cur_dim);
        //start timer
        fn(options.warm_up_n, options.n);
        //stop timer
        //print stats
  }
  return;
}

/*****************************************************************************/
void do_trmm_serial_blas(options_t options)
{
  printf("STATUS: %s.\n", __func__);
  __do_loop_and_invoke(options, __do_trmm_serial_blas);
  return;
}

void do_trmm_serial_batched(options_t options)
{
  printf("STATUS: %s.\n", __func__);
  __do_loop_and_invoke(options, __do_trmm_serial_batched);
  return;
}

void do_trmm_parallel_blas(options_t options)
{
  printf("STATUS: %s.\n", __func__);
  __do_loop_and_invoke(options, __do_trmm_parallel_blas);
  return;
}

void do_trmm_parallel_batched(options_t options)
{
  printf("STATUS: %s.\n", __func__);
  __do_loop_and_invoke(options, __do_trmm_parallel_batched);
  return;
}

#endif // KOKKOSBLAS_TRMM_PERF_TEST_H_
