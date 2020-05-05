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

//#include <complex.h>

#include "KokkosKernels_default_types.hpp"

#include <Kokkos_Random.hpp>

#include <KokkosBlas3_trmm.hpp>
#include "KokkosBatched_Trmm_Decl.hpp"
#include "KokkosBatched_Trmm_Serial_Impl.hpp"
#include "KokkosBatched_Util.hpp"

#define TRMM_PERF_TEST_DEBUG

/*************************** Print macros **************************/
#ifdef TRMM_PERF_TEST_DEBUG
#define STATUS \
 printf("STATUS: %s.\n", __func__);
#else
  #define STATUS
#endif // TRMM_PERF_TEST_DEBUG

/*************************** Test types and defaults **************************/
#define DEFAULT_TEST BLAS
#define DEFAULT_LOOP SERIAL
#define DEFAULT_MATRIX_START 10
#define DEFAULT_MATRIX_STOP 2430
#define DEFAULT_STEP 3
#define DEFAULT_WARM_UP_N 100
#define DEFAULT_N 100
#define DEFAULT_TRMM_ARGS "LUNU"
#define DEFAULT_TRMM_ALPHA 1.0
#define DEFAULT_OUT &std::cout

struct matrix_dim {
  int m, n;
};
typedef struct matrix_dim matrix_dim_t;

struct trmm_matrix_dims {
  matrix_dim_t a, b;
};
typedef struct trmm_matrix_dims trmm_matrix_dims_t;

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
  trmm_matrix_dims_t start;
  trmm_matrix_dims_t stop;
  uint32_t step;
  uint32_t warm_up_n;
  uint32_t n;
  std::string trmm_args;
  default_scalar alpha;
  std::ostream* out;
  std::string out_file;
};
typedef struct trmm_perf_test_options options_t;

using view_type_3d = Kokkos::View<default_scalar***, default_layout, default_device>;
struct trmm_args {
  char side, uplo, trans, diag;
  default_scalar alpha;
  view_type_3d A, B;
};
typedef struct trmm_args trmm_args_t;

static std::string trmm_csv_header_str = "algorithm,side-uplo-trans-diag,alpha,loop_type,A_dims,B_dims,warm_up_n,iter,total_time(s),average_time(s)";

/*************************** Internal helper fns **************************/
static void __trmm_output_csv_row(options_t options, trmm_args_t trmm_args, double time_in_seconds)
{
  options.out[0] << test_e_str[options.test] << "," <<
                  options.trmm_args << "," <<
                  options.alpha << "," <<
                  loop_e_str[options.loop] << "," <<
                  trmm_args.A.extent(1) << "x" << trmm_args.A.extent(2) << "," <<
                  trmm_args.B.extent(1) << "x" << trmm_args.B.extent(2) << "," <<
                  options.warm_up_n << "," <<
                  options.n << "," <<
                  time_in_seconds << "," <<
                  time_in_seconds / options.n << std::endl;
}

static void __print_trmm_perf_test_options(options_t options)
{
#ifdef TRMM_PERF_TEST_DEBUG
  printf("options.test      = %s\n", test_e_str[options.test].c_str());
  printf("options.loop      = %s\n", loop_e_str[options.loop].c_str());
  printf("options.start     = %dx%d,%dx%d\n", options.start.a.m, options.start.a.n, options.start.b.m, options.start.b.n);
  printf("options.stop      = %dx%d,%d,%d\n", options.stop.a.m, options.stop.a.n, options.stop.b.m, options.stop.b.n);
  printf("options.step      = %d\n", options.step);
  printf("options.warm_up_n = %d\n", options.warm_up_n);
  printf("options.n         = %d\n", options.n);
  printf("options.trmm_args = %s\n", options.trmm_args.c_str());
  printf("options.out_file  = %s\n", options.out_file.c_str());
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
#endif // TRMM_PERF_TEST_DEBUG
  return;
}

/*************************** Internal templated fns **************************/
template<class scalar_type, class vta, class vtb, class device_type>
void __do_trmm_serial_blas(options_t options, trmm_args_t trmm_args)
{
  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n = options.n;
  Kokkos::Timer timer;

  STATUS;

  for (int i = 0; i < warm_up_n; ++i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::trmm(&trmm_args.side, &trmm_args.uplo, &trmm_args.trans, 
                     &trmm_args.diag, trmm_args.alpha, A, B);
  }

  timer.reset();
  for (int i = 0; i < n ; ++i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::trmm(&trmm_args.side, &trmm_args.uplo, &trmm_args.trans, 
                     &trmm_args.diag, trmm_args.alpha, A, B);
  }
  Kokkos::fence();
  __trmm_output_csv_row(options, trmm_args, timer.seconds());
  return;
}

template<class scalar_type, class vta, class vtb, class device_type>
void __do_trmm_serial_batched(options_t options, trmm_args_t trmm_args)
{
  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n = options.n;
  Kokkos::Timer timer;
  //trmm_args.side == 'l' || trmm_args.side == 'L' ? Side::Left : 
  using side  = Side::Right;
  using uplo  = Uplo::Lower;
  using trans = Trans::NoTranspose;
  using diag  = Diag::NonUnit;
  using tag   = Algo::Trmm::Unblocked;

  STATUS;

  for (int i = 0; i < warm_up_n; ++i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    SerialTrmm<side, uplo, trans, diag, tag>::invoke(trmm_args.alpha, A, B);
  }

  timer.reset();
  for (int i = 0; i < n ; ++i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::trmm(&trmm_args.side, &trmm_args.uplo, &trmm_args.trans, 
                     &trmm_args.diag, trmm_args.alpha, A, B);
  }
  Kokkos::fence();
  __trmm_output_csv_row(options, trmm_args, timer.seconds());
  return;
}

template<class scalar_type, class vta, class vtb, class device_type>
void __do_trmm_parallel_blas(options_t options, trmm_args_t trmm_args)
{
  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n = options.n;
  Kokkos::Timer timer;

  STATUS;

  //for (int i = 0; i < warm_up_n; ++i) {
  //Kokkos::parallel_for("toLowerLoop", options.n, KOKKOS_LAMBDA (const int& i) {
  Kokkos::parallel_for("parallelBlasWarmUpLoop", warm_up_n, KOKKOS_LAMBDA (const int& i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::trmm(&trmm_args.side, &trmm_args.uplo, &trmm_args.trans, 
                     &trmm_args.diag, trmm_args.alpha, A, B);
  });

  timer.reset();
  //for (int i = 0; i < n ; ++i) {
  Kokkos::parallel_for("parallelBlasTimedLoop", n, KOKKOS_LAMBDA (const int& i) {
    auto A = Kokkos::subview(trmm_args.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(trmm_args.B, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBlas::trmm(&trmm_args.side, &trmm_args.uplo, &trmm_args.trans, 
                     &trmm_args.diag, trmm_args.alpha, A, B);
  });
  Kokkos::fence();
  __trmm_output_csv_row(options, trmm_args, timer.seconds());
  return;
}

template<class scalar_type, class vta, class vtb, class device_type>
void __do_trmm_parallel_batched(options_t options, trmm_args_t trmm_args)
{
  STATUS;
  return;
}

/*************************** Internal setup fns **************************/
template<class scalar_type, class vta, class vtb, class device_type>
trmm_args_t __do_setup(options_t options, trmm_matrix_dims_t dim)
{
  using execution_space = typename device_type::execution_space;

  trmm_args_t trmm_args;
  uint64_t seed = Kokkos::Impl::clock_tic();
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
  decltype(dim.a.m) min_dim = dim.a.m < dim.a.n ? dim.a.m : dim.a.n;
  STATUS;

  trmm_args.side  = options.trmm_args.c_str()[0];
  trmm_args.uplo  = options.trmm_args.c_str()[1];
  trmm_args.trans = options.trmm_args.c_str()[2];
  trmm_args.diag  = options.trmm_args.c_str()[3];
  trmm_args.A     = vta("trmm_args.A", options.n, dim.a.m, dim.a.n);
  trmm_args.B     = vtb("trmm_args.B", options.n, dim.b.m, dim.b.n);
  
  Kokkos::fill_random(trmm_args.A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, scalar_type>::max());
  if (trmm_args.uplo == 'U' || trmm_args.uplo == 'u') {
    // Make A upper triangular
    for (int k = 0; k < options.n; ++k) {
      auto A = Kokkos::subview(trmm_args.A, k, Kokkos::ALL(), Kokkos::ALL());
      for (int i = 1; i < dim.a.m; i++) {
        for (int j = 0; j < i; j++) {
          A(i,j) = scalar_type(0);
        }
      }
    }
  } else {
    // Make A lower triangular
    //Kokkos::parallel_for("toLowerLoop", options.n, KOKKOS_LAMBDA (const int& i) {
    for (int k = 0; k < options.n; ++k) {
      auto A = Kokkos::subview(trmm_args.A, k, Kokkos::ALL(), Kokkos::ALL());
      for (int i = 0; i < dim.a.m-1; i++) {
        for (int j = i+1; j < dim.a.n; j++) {
          A(i,j) = scalar_type(0);
        }
      }
    }
  }
  
  if (trmm_args.diag == 'U' || trmm_args.diag == 'u') {
    for (int k = 0; k < options.n; ++k) {
      auto A = Kokkos::subview(trmm_args.A, k, Kokkos::ALL(), Kokkos::ALL());
      for (int i = 0; i < min_dim; i++) {
        A(i,i) = scalar_type(1);
      }
    }
  }

  Kokkos::fill_random(trmm_args.B, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, scalar_type>::max());
  
  return trmm_args;
}

/*************************** Interal run helper fns **************************/
void __do_loop_and_invoke(options_t options, 
                          void (*fn)(options_t, trmm_args_t))
{
  trmm_matrix_dims_t cur_dims;
  trmm_args_t trmm_args;
  STATUS;

  options.out[0] << trmm_csv_header_str << std::endl;

  for (cur_dims = options.start;
        cur_dims.a.m <= options.stop.a.m && cur_dims.a.n <= options.stop.a.n &&
        cur_dims.b.m <= options.stop.b.m && cur_dims.b.n <= options.stop.b.n;
        cur_dims.a.m *= options.step, cur_dims.a.n *= options.step,
        cur_dims.b.m *= options.step, cur_dims.b.n *= options.step) {
        trmm_args = __do_setup<default_scalar, view_type_3d, view_type_3d, default_device>(options, cur_dims);
        fn(options, trmm_args);
  }
  return;
}

/*************************** External fns **************************/
void do_trmm_serial_blas(options_t options)
{ 
  STATUS;
  __do_loop_and_invoke(options, __do_trmm_serial_blas<default_scalar, view_type_3d, view_type_3d, default_device>);
  return;
}

void do_trmm_serial_batched(options_t options)
{
  STATUS;
  __do_loop_and_invoke(options, __do_trmm_serial_batched<default_scalar, view_type_3d, view_type_3d, default_device>);
  return;
}

void do_trmm_parallel_blas(options_t options)
{
  STATUS;
  __do_loop_and_invoke(options, __do_trmm_parallel_blas<default_scalar, view_type_3d, view_type_3d, default_device>);
  return;
}

void do_trmm_parallel_batched(options_t options)
{
  STATUS;
  __do_loop_and_invoke(options, __do_trmm_parallel_batched<default_scalar, view_type_3d, view_type_3d, default_device>);
  return;
}

#endif // KOKKOSBLAS_TRMM_PERF_TEST_H_
