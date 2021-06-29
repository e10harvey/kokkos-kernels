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
#ifndef KOKKOSBLAS3_GEMM_PERF_TEST_H_
#define KOKKOSBLAS3_GEMM_PERF_TEST_H_

//#include <complex.h>
#include "KokkosBlas3_common.hpp"

#include <Kokkos_Random.hpp>

#include <KokkosBlas3_gemm.hpp>

#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"
//#include "KokkosBatched_Gemm_Team_Impl.hpp"
//#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"
#include "KokkosBatched_Util.hpp"
#include "gtest/gtest.h"  // EXPECT_NEAR
#include "KokkosKernels_TestUtils.hpp"

//#define GEMM_PERF_TEST_DEBUG

////////////////////////////////////////////////////////////////////////////////
// TODOs:
// 0. set loop bounds and stride                                                [DONE in functor for multiples of 32]
// 1. Create a templated function that accepts REG_M and REG_N sizes at compile
//    time.                                                                     [DONE, functor does this now]
// 2. Check m and n size before creating functor type. Set reg_m and reg_n
//    based on m and n size.
// 3. blk_k edge case                                                           [DONE]
// 4. extended tile edge case                                                   [In Progress]
//    i.  Bounds checking - performance regression                              [Not Working]
//    ii. Using scratch memory to get results out of registers                  [Not Working]
// 5. partial tile edge case
////////////////////////////////////////////////////////////////////////////////

#if 0
// Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-example
#include <mma.h>
__device__ void wmma_ker(half *a, half *b, half *c) {
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> c_frag;

  // Initialize the output to c
  //nvcuda::wmma::fill_fragment(c_frag, c);

  // Load the inputs
  nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
  nvcuda::wmma::load_matrix_sync(b_frag, b, 16);
  nvcuda::wmma::load_matrix_sync(c_frag, c, 16, nvcuda::wmma::mem_row_major);


  // Perform the matrix multiplication
  nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

  // Store the output
  nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}
#endif

// Forward declarations
void do_gemm_serial_blas(options_t options);
void do_gemm_serial_batched(options_t options);
void do_gemm_serial_batched_blocked(options_t options);
// void do_gemm_experiment(options_t options);

// void do_gemm_serial_blas_parallel(options_t options);
// Not valid! The KokkosBlas::gemm function may take the entire device per
// invocation!
void do_gemm_serial_batched_parallel(options_t options);
void do_gemm_serial_batched_blocked_parallel(options_t options);
void do_gemm_serial_simd_batched_parallel(options_t options);
void do_gemm_serial_simd_batched_blocked_parallel(options_t options);
void do_gemm_serial_batched_compact_mkl_parallel(options_t options);
void do_gemm_team_batched_parallel(options_t options);
void do_gemm_team_batched_blocked_parallel(options_t options);
void do_gemm_team_shmem_batched_parallel(options_t options);
void do_gemm_team_shmem_batched_blocked_parallel(options_t options);
void do_gemm_team_vector_batched_parallel(options_t options);
void do_gemm_team_vector_batched_blocked_parallel(options_t options);
void do_gemm_team_simd_batched_parallel(options_t options);
void do_gemm_team_simd_batched_blocked_parallel(options_t options);
void do_gemm_experiment_parallel(options_t options);

struct SerialTag {};
struct SerialBatchDim3Tag {};
struct SerialSimdTag {};
struct SerialSimdBatchDim3Tag {};
struct TeamShmemTag {};
struct TeamShmemBatchDim3Tag {};
struct TeamTag {};
struct TeamBatchDim3Tag {};
struct TeamVectorTag {};
struct TeamVectorBatchDim3Tag {};
struct TeamSimdTag {};
struct TeamSimdBatchDim4Tag {};
struct LayoutLeftTag {};
struct LayoutRightTag {};
struct SimdCpuTag {};

void do_gemm_serial_opt1_batched_parallel(options_t options);
void do_gemm_serial_opt1_batched_blocked_parallel(options_t options);
void do_gemm_serial_opt2_batched_parallel(options_t options);
void do_gemm_serial_opt2_batched_blocked_parallel(options_t options);
void do_gemm_serial_optteam_batched_parallel(options_t options);
void do_gemm_serial_optteam_batched_blocked_parallel(options_t options);
void do_gemm_team_optdivisor_batched_parallel(options_t options);
void do_gemm_team_optdivisor_batched_blocked_parallel(options_t options);
void do_gemm_team_opt1_batched_parallel(options_t options);
void do_gemm_team_opt1_batched_blocked_parallel(options_t options);

// Optimization level tags
// Opt1 is level 1: increase the number of threads by factor 'columns of C'
// Opt2 is level 2: increase the number of threads by factor 'columns of C *
// rows of C' OptDivisor: decrease the number of threads by factor 'divisor'
// OptTeam: Use serial gemm inside a team of threads
struct SerialTagOpt1 {};
struct SerialBatchDim3TagOpt1 {};
struct SerialTagOpt2 {};
struct SerialBatchDim3TagOpt2 {};
struct SerialTagOpt2Tiled {};
struct SerialBatchDim3TagOpt2Tiled {};
struct SerialTagOptTeam {};
struct SerialBatchDim3TagOptTeam {};
struct TeamTagOptDivisor {};
struct TeamBatchDim3TagOptDivisor {};
struct TeamTagOpt1 {};
struct TeamBatchDim3TagOpt1 {};

// gemm invoke table
void (*do_gemm_invoke[LOOP_N][TEST_N])(options_t) = {
    {
        do_gemm_serial_blas,                                     // BLAS
        do_gemm_serial_batched, do_gemm_serial_batched_blocked,  // Serial
        NULL, NULL,                                              // Serial Opt1
        NULL, NULL,                                              // Serial SIMD
        NULL,        // Serial Compact MKL
        NULL, NULL,  // Team
        NULL, NULL,  // TeamVector
        NULL, NULL,  // TeamSimd
        NULL         // Serial Experiment
    },
    {
        NULL,                             // BLAS
        do_gemm_serial_batched_parallel,  // Serial
        do_gemm_serial_batched_blocked_parallel,
        do_gemm_serial_opt1_batched_parallel,  // Serial Opt1
        do_gemm_serial_opt1_batched_blocked_parallel,
        do_gemm_serial_opt2_batched_parallel,  // Serial Opt2
        do_gemm_serial_opt2_batched_blocked_parallel,
        do_gemm_serial_optteam_batched_parallel,  // Serial OptTeam
        do_gemm_serial_optteam_batched_blocked_parallel,
        do_gemm_serial_simd_batched_parallel,  // Serial SIMD
        do_gemm_serial_simd_batched_blocked_parallel,
        do_gemm_serial_batched_compact_mkl_parallel,  // Serial MKL
        do_gemm_team_batched_parallel,                // Team
        do_gemm_team_batched_blocked_parallel,
        do_gemm_team_shmem_batched_parallel,                // Team_Shmem
        do_gemm_team_shmem_batched_blocked_parallel,
        do_gemm_team_opt1_batched_parallel,
        do_gemm_team_opt1_batched_blocked_parallel,
        do_gemm_team_optdivisor_batched_blocked_parallel,  // Team OptDivisor
        do_gemm_team_optdivisor_batched_parallel,
        do_gemm_team_vector_batched_parallel,
        NULL,                                // TeamVector
        do_gemm_team_simd_batched_parallel,  // TeamSimd
        do_gemm_team_simd_batched_blocked_parallel,
        do_gemm_experiment_parallel  // Parallel Experiment
    }};

/*************************** Test types and defaults **************************/
#define DEFAULT_GEMM_ARGS "NN"
#define DEFAULT_GEMM_ALPHA 1.0
#define DEFAULT_GEMM_BETA 1.0

using view_type_2d_scratch = Kokkos::View<default_scalar **, default_layout, typename default_device::scratch_memory_space>;
using view_type_2d_scratch_layout_left = Kokkos::View<default_scalar **, Kokkos::LayoutLeft, typename default_device::scratch_memory_space>;

using view_type_3d =
    Kokkos::View<default_scalar ***, default_layout, default_device>;
using view_type_4d =
    Kokkos::View<default_scalar ****, default_layout, default_device>;
using view_type_5d =
    Kokkos::View<default_scalar *****, default_layout, default_device>;

// Construct the vector type
using memory_space = typename default_device::execution_space::memory_space;
constexpr int simd_vector_size =
    KokkosBatched::DefaultVectorLength<default_scalar, memory_space>::value;
constexpr int simd_internal_vector_size =
    KokkosBatched::DefaultInternalVectorLength<default_scalar,
                                               memory_space>::value;
using vector_type = KokkosBatched::Vector<KokkosBatched::SIMD<default_scalar>,
                                          simd_vector_size>;
using internal_vector_type =
    KokkosBatched::Vector<KokkosBatched::SIMD<default_scalar>,
                          simd_internal_vector_size>;
using vector_view_type_3d =
    Kokkos::View<vector_type ***, default_layout, default_device>;
using internal_vector_view_type_4d =
    Kokkos::View<internal_vector_type ****, default_layout, default_device>;

struct batched_params {
  int team_size;
  int vector_len;
  int divisor;
};
typedef struct batched_params batched_params_t;

/**
 * @brief struct gemm_simd_args encapsulates the data types required
 * for allocating and passing a single matrix to the KokkosBatched gemm
 * kernels. To invoke gemm on a batch of matrices, three instances of this
 * struct are required, one for each matrix, A, B, and C.
 *
 * @var  vec_3d: 3-rank view type used for allocating the underlying data.
 *               A reference must be kept to this object to ensure the
 *               data is not free'd by the C++ runtime.
 * @var  mat_4d: 4-rank view type used for populating the simd view with
                 random values.
 * @var ivec_4d: 4-rank view type used for passing to math kernels. This
 *               view type is used for leveraging simd instructions on
 *               both the host and device.
 */
struct gemm_simd_args {
  vector_view_type_3d vec_3d;
  view_type_4d mat_4d;
  internal_vector_view_type_4d ivec_4d;
};
typedef struct gemm_simd_args gemm_simd_args_t;

/**
 * @brief struct gemm_args are common arguments passed to
 * both gemm implementations in the KokkosBlas and KokkosBatched
 * namespaces throughout these performance tests.
 *
 * @var transA: transpose type for A matrix.
 *              supported types:   'n' - no transpose, 't' - transpose.
 *              unsupported types: 'c' - conjugate transpose.
 * @var transB: transpose type for B matrix.
 *              supported types:   'n' - no transpose, 't' - transpose.
 *              unsupported types: 'c' - conjugate transpose.
 * @var alpha: scalar applied to A matrix.
 * @var beta:  scalar applied to B matrix.
 * @var A:     3-rank view type used in all non-simd tests.
 * @var B:     3-rank view type used in all non-simd tests.
 * @var C:     3-rank view type used in all non-simd tests.
 * @var bp:    team_size and vector_length for tests that use
 * Kokkos::TeamPolicy.
 * @var Av:    3-rank and 4-rank vector view types for simd tests.
 * @var Bv:    3-rank and 4-rank vector view types for simd tests.
 * @var Cv:    3-rank and 4-rank vector view types for simd tests.
 */
struct gemm_args {
  char transA, transB;
  default_scalar alpha;
  default_scalar beta;
  view_type_3d A, B, C;
  batched_params_t bp;
  // Below are matrices for simd tests
  gemm_simd_args_t Av, Bv, Cv;
  matrix_dims_t dims;
};
typedef struct gemm_args gemm_args_t;

static std::string gemm_csv_header_str =
    "algorithm,vector_type,transAtransB,alpha,beta,team_size,vector_len,loop_"
    "type,A_dims,B_"
    "dims,C_dims,warm_up_n,"
    "iter,total_time(s),average_time(s),FLOPS,GFLOP/average_time(s)";

/*************************** Internal helper fns **************************/
// Flop count formula from lapack working note 41:
// http://www.icl.utk.edu/~mgates3/docs/lawn41.pdf
static inline double __gemm_flop_count(double a_m, double a_n, double b_n) {
  if (std::is_same<double, default_scalar>::value ||
      std::is_same<float, default_scalar>::value ||
      std::is_same<Kokkos::Experimental::half_t, default_scalar>::value)
    return 2 * a_m * b_n * a_n;
  else
    // For complex, we need to count 2 flops for each add and 6 flops for each
    // multiply.
    return (2 + 6) * a_m * b_n * a_n;
}

static inline std::string __gemm_output_dim_string(options_t options,
                                                   matrix_dim_t dim) {
  std::string x   = "x";
  std::string ret = std::to_string(dim.m) + x + std::to_string(dim.n);

  if (options.blas_args.batch_size_last_dim)
    return ret + x + std::to_string(dim.k);
  else
    return std::to_string(dim.k) + x + ret;
}

static void __gemm_output_csv_row(options_t options, gemm_args_t gemm_args,
                                  double time_in_seconds,
                                  const char *experiment_name = nullptr) {
  std::string algo_name = test_e_str[options.test];
  std::string ts        = std::to_string(gemm_args.bp.team_size);
  std::string vlen      = std::to_string(gemm_args.bp.vector_len);
  std::string vtype     = internal_vector_type::label();
  if (experiment_name) algo_name = std::string(experiment_name);
  if (options.blas_args.use_auto) ts = vlen = "Kokkos::AUTO";

  double flops;
  double gflops;
  double average_time = time_in_seconds / options.n;

  if (options.verify) return;

  flops = gemm_args.dims.a.k * __gemm_flop_count(gemm_args.dims.a.m,
                                                 gemm_args.dims.a.n,
                                                 gemm_args.dims.b.n);

  gflops = flops / 1e9;

  options.out[0] << algo_name << "," << vtype << ","
                 << options.blas_args.gemm.gemm_args << ","
                 << static_cast<double>(options.blas_args.gemm.alpha) << ","
                 << static_cast<double>(options.blas_args.gemm.beta) << ","
                 << ts << "," << vlen << "," << loop_e_str[options.loop] << ","
                 << __gemm_output_dim_string(options, gemm_args.dims.a) << ","
                 << __gemm_output_dim_string(options, gemm_args.dims.b) << ","
                 << __gemm_output_dim_string(options, gemm_args.dims.c) << ","
                 << options.warm_up_n << "," << options.n << ","
                 << time_in_seconds << "," << time_in_seconds / options.n << ","
                 << flops << "," << gflops / average_time << std::endl;
}

static void __print_gemm_perf_test_options(options_t options) {
#ifdef PERF_TEST_DEBUG
  printf("options.test      = %s\n", test_e_str[options.test].c_str());
  printf("options.loop      = %s\n", loop_e_str[options.loop].c_str());
  printf("options.start     = %dx%d,%dx%d\n", options.start.a.m,
         options.start.a.n, options.start.b.m, options.start.b.n);
  printf("options.stop      = %dx%d,%dx%d\n", options.stop.a.m,
         options.stop.a.n, options.stop.b.m, options.stop.b.n);
  printf("options.step      = %d\n", options.step);
  printf("options.warm_up_n = %d\n", options.warm_up_n);
  printf("options.n         = %d\n", options.n);
  printf("options.blas_args.gemm.gemm_args = %s\n",
         options.blas_args.gemm.gemm_args.c_str());
  printf("options.out_file  = %s\n", options.out_file.c_str());
  if (std::is_same<double, default_scalar>::value)
    printf("options.alpha     = %lf\n", options.blas_args.gemm.alpha);
  else if (std::is_same<float, default_scalar>::value)
    printf("options.alpha     = %f\n", options.blas_args.gemm.alpha);
#endif  // PERF_TEST_DEBUG
  return;
}

/*************************** Internal templated fns **************************/
template <class scalar_type, class vta, class vtb, class device_type>
void __do_gemm_serial_blas(options_t options, gemm_args_t gemm_args) {
// Need to take subviews on the device
#if !defined(KOKKOS_ENABLE_CUDA)
  Kokkos::Timer timer;

  STATUS;

  auto __do_loop = [](uint32_t n, gemm_args_t _gemm_args,
                      bool batch_size_last_dim) {
    for (uint32_t i = 0; i < n; ++i) {
      for (int j = 0; j < _gemm_args.dims.c.k; j++) {
        auto A = Kokkos::subview(_gemm_args.A, j, Kokkos::ALL(), Kokkos::ALL());
        auto B = Kokkos::subview(_gemm_args.B, j, Kokkos::ALL(), Kokkos::ALL());
        auto C = Kokkos::subview(_gemm_args.C, j, Kokkos::ALL(), Kokkos::ALL());
        if (batch_size_last_dim) {
          A = Kokkos::subview(_gemm_args.A, Kokkos::ALL(), Kokkos::ALL(), j);
          B = Kokkos::subview(_gemm_args.B, Kokkos::ALL(), Kokkos::ALL(), j);
          C = Kokkos::subview(_gemm_args.C, Kokkos::ALL(), Kokkos::ALL(), j);
        }

        KokkosBlas::gemm(&_gemm_args.transA, &_gemm_args.transB,
                         _gemm_args.alpha, A, B, _gemm_args.beta, C);
      }
    }
  };
  __do_loop(options.warm_up_n, gemm_args,
            options.blas_args.batch_size_last_dim);
  Kokkos::fence();

  timer.reset();
  __do_loop(options.n, gemm_args, options.blas_args.batch_size_last_dim);
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds());
#else
  std::cerr << std::string(__func__)
            << " disabled since KOKKOS_ENABLE_CUDA is defined." << std::endl;
#endif  // !KOKKOS_ENABLE_CUDA
  return;
}

template <class TransAType, class TransBType, class AlgoType>
void __do_gemm_serial_batched_template(options_t options,
                                       gemm_args_t gemm_args) {
// Need to take subviews on the device
#if !defined(KOKKOS_ENABLE_CUDA)
  Kokkos::Timer timer;

  auto __do_loop = [](uint32_t n, gemm_args_t _gemm_args,
                      bool batch_size_last_dim) {
    for (uint32_t i = 0; i < n; ++i) {
      for (int j = 0; j < _gemm_args.dims.c.k; j++) {
        auto A = Kokkos::subview(_gemm_args.A, j, Kokkos::ALL(), Kokkos::ALL());
        auto B = Kokkos::subview(_gemm_args.B, j, Kokkos::ALL(), Kokkos::ALL());
        auto C = Kokkos::subview(_gemm_args.C, j, Kokkos::ALL(), Kokkos::ALL());
        if (batch_size_last_dim) {
          A = Kokkos::subview(_gemm_args.A, Kokkos::ALL(), Kokkos::ALL(), j);
          B = Kokkos::subview(_gemm_args.B, Kokkos::ALL(), Kokkos::ALL(), j);
          C = Kokkos::subview(_gemm_args.C, Kokkos::ALL(), Kokkos::ALL(), j);
        }

        SerialGemm<TransAType, TransBType, AlgoType>::invoke(
            _gemm_args.alpha, A, B, _gemm_args.beta, C);
      }
    }
  };

  __do_loop(options.warm_up_n, gemm_args,
            options.blas_args.batch_size_last_dim);
  Kokkos::fence();

  timer.reset();
  __do_loop(options.n, gemm_args, options.blas_args.batch_size_last_dim);
  Kokkos::fence();
  __gemm_output_csv_row(options, gemm_args, timer.seconds());
#else
  std::cerr << std::string(__func__)
            << " disabled since KOKKOS_ENABLE_CUDA is defined." << std::endl;
#endif  // !KOKKOS_ENABLE_CUDA
}

template <class scalar_type, class vta, class vtb, class vtc, class device_type,
          class algo_type>
void __do_gemm_serial_batched(options_t options, gemm_args_t gemm_args) {
  char a  = toupper(gemm_args.transA);
  char b  = toupper(gemm_args.transB);
  using N = Trans::NoTranspose;
  using T = Trans::Transpose;
  // using C = Trans::ConjTranspose;

  STATUS;

  if (a == 'N' && b == 'N') {
    __do_gemm_serial_batched_template<N, N, algo_type>(options, gemm_args);
  } else if (a == 'N' && b == 'T') {
    __do_gemm_serial_batched_template<N, T, algo_type>(options, gemm_args);
    //} else if (a == 'N' && b == 'C') {
    //  __do_gemm_serial_batched_template<N, C, algo_type>(options, gemm_args);
  } else if (a == 'T' && b == 'N') {
    __do_gemm_serial_batched_template<T, N, algo_type>(options, gemm_args);
  } else if (a == 'T' && b == 'T') {
    __do_gemm_serial_batched_template<T, T, algo_type>(options, gemm_args);
    //} else if (a == 'T' && b == 'C') {
    //  __do_gemm_serial_batched_template<T, C, algo_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'N') {
    //  __do_gemm_serial_batched_template<C, N, algo_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'T') {
    //  __do_gemm_serial_batched_template<C, T, algo_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'C') {
    //  __do_gemm_serial_batched_template<C, C, algo_type>(options, gemm_args);
  } else {
    FATAL_ERROR("Bad gemm_args TransA or TransB value");
  }
  return;
}

template <class TransAType, class TransBType, class BlockingType>
struct parallel_batched_gemm_range_policy {
  gemm_args_t gemm_args_;
  // The divisor is used in optimized operators, which tags containing
  // "Opt". The divisor is used to decrease the size of the submatrices
  // passed to each gemm.
  size_t divisor_,
      // 2rank C matrix rows (m) and cols (n)
      c_m_, c_n_,
      // tiles_per_c_m_: Number of tiles per every row of a 2rank C matrix
      // tiles_per_c_n_: Number of tiles per every col of a 2rank C matrix
      // tiles_per_2rank_matrix: Number of tiles per every 2rank C matrix
      tiles_per_c_m_, tiles_per_c_n_, tiles_per_2rank_matrix_,
      // tile_m_: 2rank tile rows (m)
      // tile_n_: 2rank tile cols (n)
      tile_m_, tile_n_, tile_mn_, eles_per_tile_row_;

  parallel_batched_gemm_range_policy(gemm_args_t gemm_args,
                                     bool batch_size_last_dim,
                                     size_t divisor = 1, size_t tile_m = 1,
                                     size_t tile_n = 1)
      : gemm_args_(gemm_args),
        divisor_(divisor),
        tile_m_(tile_m),
        tile_n_(tile_n) {
    if (batch_size_last_dim) {
      c_m_ = gemm_args_.C.extent(0);
      c_n_ = gemm_args_.C.extent(1);
    } else {
      c_m_ = gemm_args_.C.extent(1);
      c_n_ = gemm_args_.C.extent(2);
    }
    tiles_per_c_m_          = c_m_ / tile_m_;
    tiles_per_c_n_          = c_n_ / tile_n_;
    tiles_per_2rank_matrix_ = tiles_per_c_m_ * tiles_per_c_n_;
    tile_mn_                = tile_m_ * tile_n_;
    eles_per_tile_row_      = tile_mn_ * tiles_per_c_n_;
  }

  //__device__
  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTag &, const int &i) const {
    auto svA = Kokkos::subview(gemm_args_.A, i, Kokkos::ALL(),
                               Kokkos::ALL());  // m x k = 16 x 16
    auto svB = Kokkos::subview(gemm_args_.B, i, Kokkos::ALL(),
                               Kokkos::ALL());  // k x n = 16 x 16
    auto svC = Kokkos::subview(gemm_args_.C, i, Kokkos::ALL(),
                               Kokkos::ALL());  // m x n = 16 x 16

    // wmma_ker((half *) svA.data(), (half *) svB.data(), (half *) svC.data());
    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3Tag &, const int &i) const {
    auto svA = Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svB = Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svC = Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), i);

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt1 &, const int &i) const {
    // Select next matrix everytime i is a new multiple of divisor
    auto batch_idx = i / divisor_;
    // Select col of B and C
    auto col_idx = i % divisor_;

    auto svA =
        Kokkos::subview(gemm_args_.A, batch_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svB_col =
        Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), col_idx);
    auto svC_col =
        Kokkos::subview(gemm_args_.C, batch_idx, Kokkos::ALL(), col_idx);

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB_col, gemm_args_.beta, svC_col);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt1 &, const int &i) const {
    // Select next matrix everytime i is a new multiple of divisor
    auto batch_idx = i / divisor_;
    // Select col of B and C
    auto col_idx = i % divisor_;

    auto svA =
        Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), batch_idx);
    auto svB_col =
        Kokkos::subview(gemm_args_.B, Kokkos::ALL(), col_idx, batch_idx);
    auto svC_col =
        Kokkos::subview(gemm_args_.C, Kokkos::ALL(), col_idx, batch_idx);

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB_col, gemm_args_.beta, svC_col);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt2 &, const int &i) const {
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor_;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor_;  // ex: 2x2 -- 0,1,2,3
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % gemm_args_.C.extent(2);  // ex: 2x2 -- 0,1,0,1
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / gemm_args_.C.extent(1);  // ex: 2x2 -- 0,0,1,1

    auto svA_row =
        Kokkos::subview(gemm_args_.A, batch_idx, row_idx, Kokkos::ALL());
    auto svB_col =
        Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), col_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, batch_idx, row_idx, col_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt2 &, const int &i) const {
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor_;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor_;  // ex: 2x2 -- 0,1,2,3
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % gemm_args_.C.extent(1);  // ex: 2x2 -- 0,1,0,1
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / gemm_args_.C.extent(0);  // ex: 2x2 -- 0,0,1,1

    auto svA_row =
        Kokkos::subview(gemm_args_.A, row_idx, Kokkos::ALL(), batch_idx);
    auto svB_col =
        Kokkos::subview(gemm_args_.B, Kokkos::ALL(), col_idx, batch_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, row_idx, col_idx, batch_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt2Tiled &, const int &i) const {
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor_;

    // For each thread, compute the given tile's row index, this spans the tile
    // size (tile_mn_) by the number of tiles that fit in our
    // columns (tiles_per_c_n_)
    auto tile_m_idx = (i % divisor_) / eles_per_tile_row_;
    // For each thread, compute the given tile's columns index, this is always
    // within [0, tiles_per_c_n_) but each thread must find its own column index
    auto tile_n_idx = (i / tile_mn_) % tiles_per_c_n_;

    // For every batch, we need mod in [0, tile_mn_)
    auto mod = i % tile_mn_;
    // For every mod, we need a column index in [0, c_n_). Since tiles are used
    // for thread work assignment, we must stride the tile patterns of [0,
    // tiles_per_c_n_) by tile_n_idx * tile_n_ to find the correct offset within
    // the given tile of C/A/B.
    auto col_idx = (mod % tile_n_) + tile_n_idx * tile_n_;
    // For every mod, we need a row index in [0, c_m_). Since tiles are used
    // for thread work assignment, we must stride the tile patterns of [0,
    // tiles_per_c_m_) by tile_m_idx * tile_m_ to find the correct offset within
    // the given tile of C/A/B.
    auto row_idx = (mod / tile_n_) + tile_m_idx * tile_m_;
    /*printf("dim1, i:%d,C(%lu,%lu),tile_m_idx:%lu,tile_n_idx:%lu,mod:%lu\n", i,
           row_idx, col_idx, tile_m_idx, tile_n_idx, mod);*/

    auto svA_row =
        Kokkos::subview(gemm_args_.A, batch_idx, row_idx, Kokkos::ALL());
    auto svB_col =
        Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), col_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, batch_idx, row_idx, col_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt2Tiled &, const int &i) const {
    auto batch_idx = i / divisor_;

    // For each thread, compute the given tile's row index, this spans the tile
    // size (tile_mn_) by the number of tiles that fit in our
    // columns (tiles_per_c_n_)
    auto tile_m_idx = (i % divisor_) / eles_per_tile_row_;
    // For each thread, compute the given tile's columns index, this is always
    // within [0, tiles_per_c_n_) but each thread must find its own column index
    auto tile_n_idx = (i / tile_mn_) % tiles_per_c_n_;

    // For every batch, we need mod in [0, tile_mn_)
    auto mod = i % tile_mn_;
    // For every mod, we need a column index in [0, c_n_). Since tiles are used
    // for thread work assignment, we must stride the tile patterns of [0,
    // tiles_per_c_n_) by tile_n_idx * tile_n_ to find the correct offset within
    // the given tile of C/A/B.
    auto col_idx = (mod % tile_n_) + tile_n_idx * tile_n_;
    // For every mod, we need a row index in [0, c_m_). Since tiles are used
    // for thread work assignment, we must stride the tile patterns of [0,
    // tiles_per_c_m_) by tile_m_idx * tile_m_ to find the correct offset within
    // the given tile of C/A/B.
    auto row_idx = (mod / tile_n_) + tile_m_idx * tile_m_;
    /*printf("dim1, i:%d,C(%lu,%lu),tile_m_idx:%lu,tile_n_idx:%lu,mod:%lu\n", i,
           row_idx, col_idx, tile_m_idx, tile_n_idx, mod);*/

    auto svA_row =
        Kokkos::subview(gemm_args_.A, row_idx, Kokkos::ALL(), batch_idx);
    auto svB_col =
        Kokkos::subview(gemm_args_.B, Kokkos::ALL(), col_idx, batch_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, row_idx, col_idx, batch_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialSimdTag &, const int &i) const {
    auto svA =
        Kokkos::subview(gemm_args_.Av.vec_3d, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB =
        Kokkos::subview(gemm_args_.Bv.vec_3d, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC =
        Kokkos::subview(gemm_args_.Cv.vec_3d, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialSimdBatchDim3Tag &, const int &i) const {
    auto svA =
        Kokkos::subview(gemm_args_.Av.vec_3d, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svB =
        Kokkos::subview(gemm_args_.Bv.vec_3d, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svC =
        Kokkos::subview(gemm_args_.Cv.vec_3d, Kokkos::ALL(), Kokkos::ALL(), i);

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOptTeam &, const int &i) const {
    Kokkos::abort("SerialTagOptTeam not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOptTeam &, const int &i) const {
    Kokkos::abort("SerialBatchDim3TagOptTeam not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTag &, const int &i) const {
    Kokkos::abort("TeamTag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3Tag &, const int &i) const {
    Kokkos::abort("TeamBatchDim3Tag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamShmemTag &, const int &i) const {
    Kokkos::abort("TeamShmemTag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamShmemBatchDim3Tag &, const int &i) const {
    Kokkos::abort("TeamShmemBatchDim3Tag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTagOpt1 &, const int &i) const {
    Kokkos::abort("TeamTagOpt1 not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3TagOpt1 &, const int &i) const {
    Kokkos::abort("TeamBatchDim3TagOpt1 not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTagOptDivisor &, const int &i) const {
    Kokkos::abort("TeamTagOptDivisor not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3TagOptDivisor &, const int &i) const {
    Kokkos::abort(
        "TeamBatchDim3TagOptDivisor not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamVectorTag &, const int &i) const {
    Kokkos::abort("TeamVectorTag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamVectorBatchDim3Tag &, const int &i) const {
    Kokkos::abort("TeamVectorBatchDim3Tag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamSimdTag &, const int &i) const {
    Kokkos::abort("TeamSimdTag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamSimdBatchDim4Tag &, const int &i) const {
    Kokkos::abort("TeamSimdBatchDim4Tag not supported using RangePolicy.");
  }
};

template <class MemberType, class TransAType, class TransBType,
          class BlockingType,
          int REG_M, int REG_N, int STRIDE_M, int STRIDE_N,
          class AlgoMode = void>
struct parallel_batched_gemm {
  gemm_args_t gemm_args_;
  size_t divisor_, n_sub_blocks, n_blk_k_blocks;
  unsigned tiles_per_row, tiles_per_col, blk_m, blk_n, blk_k;

  parallel_batched_gemm(gemm_args_t gemm_args, bool batch_size_last_dim, size_t divisor = 1, unsigned tile_m = 1, unsigned tile_n = 1, unsigned tile_k = 1)
    : gemm_args_(gemm_args), divisor_(divisor), blk_m(tile_m), blk_n(tile_n), blk_k(tile_k) {
    if (batch_size_last_dim) {
      tiles_per_row = (unsigned)gemm_args.C.extent(0) / blk_m + !!(gemm_args.C.extent(0) % blk_m);
      tiles_per_col = (unsigned)gemm_args.C.extent(1) / blk_n + !!(gemm_args.C.extent(1) % blk_n);
      n_blk_k_blocks = (unsigned)gemm_args.A.extent(1) / blk_k + !!(gemm_args.A.extent(1) % blk_k);
    } else {
      tiles_per_row = (unsigned)gemm_args.C.extent(1) / blk_m + !!(gemm_args.C.extent(1) % blk_m);
      tiles_per_col = (unsigned)gemm_args.C.extent(2) / blk_n + !!(gemm_args.C.extent(2) % blk_n);
      n_blk_k_blocks = (unsigned)gemm_args.A.extent(2) / blk_k + !!(gemm_args.A.extent(2) % blk_k);
    }
    n_sub_blocks = tiles_per_row * tiles_per_col;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(gemm_args_.B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(gemm_args_.C, i, Kokkos::ALL(), Kokkos::ALL());

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3Tag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svB = Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svC = Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), i);

    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }

  // TODO: Why is TeamTagOpt1 incorrect?
  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTagOpt1 &, const MemberType &member) const {
    auto i = member.league_rank();
    // Select next matrix everytime i is a new multiple of divisor
    auto batch_idx = i / divisor_;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, divisor_), [&](const int &col_idx) {
          // Select col of B and C
          auto svA = Kokkos::subview(gemm_args_.A, batch_idx, Kokkos::ALL(),
                                     Kokkos::ALL());
          auto svB_col =
              Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), col_idx);
          auto svC_col =
              Kokkos::subview(gemm_args_.C, batch_idx, Kokkos::ALL(), col_idx);

          KokkosBatched::SerialGemm<TransAType, TransBType,
                                    BlockingType>::invoke(gemm_args_.alpha, svA,
                                                          svB_col,
                                                          gemm_args_.beta,
                                                          svC_col);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3TagOpt1 &,
                  const MemberType &member) const {
    auto i = member.league_rank();
    // Select next matrix everytime i is a new multiple of divisor
    auto batch_idx = i / divisor_;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, divisor_), [&](const int &col_idx) {
          // Select col of B and C
          auto svA = Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(),
                                     batch_idx);
          auto svB_col =
              Kokkos::subview(gemm_args_.B, Kokkos::ALL(), col_idx, batch_idx);
          auto svC_col =
              Kokkos::subview(gemm_args_.C, Kokkos::ALL(), col_idx, batch_idx);

          KokkosBatched::SerialGemm<TransAType, TransBType,
                                    BlockingType>::invoke(gemm_args_.alpha, svA,
                                                          svB_col,
                                                          gemm_args_.beta,
                                                          svC_col);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt2 &, const MemberType &member) const {
    auto i = member.league_rank();
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor_;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor_;  // ex: 2x2 -- 0,1,2,3
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % gemm_args_.C.extent(2);  // ex: 2x2 -- 0,1,0,1
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / gemm_args_.C.extent(1);  // ex: 2x2 -- 0,0,1,1

    auto svA_row =
        Kokkos::subview(gemm_args_.A, batch_idx, row_idx, Kokkos::ALL());
    auto svB_col =
        Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), col_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, batch_idx, row_idx, col_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt2 &,
                  const MemberType &member) const {
    auto i = member.league_rank();
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor_;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor_;  // ex: 2x2 -- 0,1,2,3
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % gemm_args_.C.extent(1);  // ex: 2x2 -- 0,1,0,1
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / gemm_args_.C.extent(0);  // ex: 2x2 -- 0,0,1,1

    auto svA_row =
        Kokkos::subview(gemm_args_.A, row_idx, Kokkos::ALL(), batch_idx);
    auto svB_col =
        Kokkos::subview(gemm_args_.B, Kokkos::ALL(), col_idx, batch_idx);
    auto svC_ele = Kokkos::subview(gemm_args_.C, row_idx, col_idx, batch_idx);

    // TODO: Fix subview for svA_row and add back in TransAType.
    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
                              BlockingType>::invoke(gemm_args_.alpha, svA_row,
                                                    svB_col, gemm_args_.beta,
                                                    svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOptTeam &, const MemberType &member) const {
    auto i = member.league_rank();
    auto start = i * divisor_;
    auto end   = start + divisor_;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, start, end),
        [&](const int &j) {
          auto svA =
              Kokkos::subview(gemm_args_.A, j, Kokkos::ALL(), Kokkos::ALL());
          auto svB =
              Kokkos::subview(gemm_args_.B, j, Kokkos::ALL(), Kokkos::ALL());
          auto svC =
              Kokkos::subview(gemm_args_.C, j, Kokkos::ALL(), Kokkos::ALL());

          KokkosBatched::SerialGemm<TransAType, TransBType,
                                    BlockingType>::invoke(gemm_args_.alpha, svA,
                                                          svB, gemm_args_.beta,
                                                          svC);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOptTeam &,
                  const MemberType &member) const {
    auto i = member.league_rank();
    auto start = i * divisor_; // 0, 2, 4
    auto end   = start + divisor_; // 2, 4, 6

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, start, end),
        [&](const int &j) {
          auto svA =
              Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), j);
          auto svB =
              Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), j);
          auto svC =
              Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), j);

          KokkosBatched::SerialGemm<TransAType, TransBType,
                                    BlockingType>::invoke(gemm_args_.alpha, svA,
                                                          svB, gemm_args_.beta,
                                                          svC);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamShmemTag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(gemm_args_.B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(gemm_args_.C, i, Kokkos::ALL(), Kokkos::ALL());

    view_type_2d_scratch svA_scr(member.team_scratch(0), svA.extent(0),  svA.extent(1));
    view_type_2d_scratch svB_scr(member.team_scratch(0), svB.extent(0),  svB.extent(1));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svB.extent(0)), [&](const int &i) {
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svB.extent(1)), [&](const int &j) {
	    svB_scr(i,j) = svB(i,j); // TODO: reduce bank conflicts
	  });
      });

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svA.extent(0)), [&](const int &i) {
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svA.extent(1)), [&](const int &j) {
	    svA_scr(i,j) = svA(i,j); // TODO: reduce bank conflicts
	  });
      });

    // Wait for A and B to reside in scratch memory
    member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svC.extent(0)),[&](const int &row_idx) {
	auto svA_row = Kokkos::subview(svA_scr, row_idx, Kokkos::ALL());
	// DONE: reduce scratch size and lazy copy svA_row -- svA_row_scr = svA_row: Doesn't work when team_size > 1.
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svC.extent(1)),[&](const int &col_idx) {
	    auto svB_col = Kokkos::subview(svB_scr, Kokkos::ALL(), col_idx); //256 bytes apart
	    auto svC_ele = Kokkos::subview(svC, row_idx, col_idx);

	    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
				      BlockingType>::invoke(gemm_args_.alpha,
							    svA_row,
							    svB_col,
							    gemm_args_.beta,
							    svC_ele);
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamShmemBatchDim3Tag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svB = Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svC = Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), i);

    view_type_2d_scratch svA_scr(member.team_scratch(0), svA.extent(0),  svA.extent(1));
    view_type_2d_scratch svB_scr(member.team_scratch(0), svB.extent(0),  svB.extent(1));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svB.extent(0)), [&](const int &i) {
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svB.extent(1)), [&](const int &j) {
	    svB_scr(i,j) = svB(i,j); // TODO: reduce bank conflicts
	  });
      });

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svA.extent(0)), [&](const int &i) {
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svA.extent(1)), [&](const int &j) {
	    svA_scr(i,j) = svA(i,j); // TODO: reduce bank conflicts
	  });
      });

    // Wait for A and B to reside in scratch memory
    member.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svC.extent(0)),[&](const int &row_idx) {
	auto svA_row = Kokkos::subview(svA_scr, row_idx, Kokkos::ALL());
	// DONE: reduce scratch size and lazy copy svA_row -- svA_row_scr = svA_row: Doesn't work when team_size > 1.
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svC.extent(1)),[&](const int &col_idx) {
	    auto svB_col = Kokkos::subview(svB_scr, Kokkos::ALL(), col_idx); //256 bytes apart
	    auto svC_ele = Kokkos::subview(svC, row_idx, col_idx);

	    KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
				      BlockingType>::invoke(gemm_args_.alpha,
							    svA_row,
							    svB_col,
							    gemm_args_.beta,
							    svC_ele);
	  });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTag &, const MemberType &member) const {
    default_scalar reg_c[REG_M][REG_N] = { {0} };
    unsigned batch_idx = member.league_rank() / n_sub_blocks;

    auto svA = Kokkos::subview(gemm_args_.A, batch_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(gemm_args_.B, batch_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(gemm_args_.C, batch_idx, Kokkos::ALL(), Kokkos::ALL());

    {
    default_scalar prefetch_reg_a[REG_M], prefetch_reg_b[REG_N];
    default_scalar reg_a[REG_M], reg_b[REG_N];

    unsigned local_team_idx = member.league_rank() % n_sub_blocks;
    unsigned start_m = (local_team_idx / tiles_per_col) * blk_m;
    unsigned start_n = (local_team_idx % tiles_per_col) * blk_n;

    // intentionally allocate svA_scr transposed, its faster to perform transpose during shmem population
    view_type_2d_scratch svA_scr(member.team_scratch(0), blk_k, blk_m);
    view_type_2d_scratch svB_scr(member.team_scratch(0), blk_k, blk_n);

    // Here, we populate scratch memory with one or more blk_k for every thread of the team!
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_n / REG_N), [&](const int &thread_id) {
        auto thread_offset = thread_id + start_n;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
            for (int i = 0; i < REG_N * STRIDE_N; i+= STRIDE_N)
              svB_scr(vlane_id, thread_id + i) = svB(vlane_id, thread_offset + i);
          });
      });

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
        auto thread_offset = thread_id + start_m;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
            for (int i = 0; i < REG_M * STRIDE_M; i+= STRIDE_M)
              svA_scr(vlane_id, thread_id + i) = svA(thread_offset + i, vlane_id);
          });
      });

    // Check whether we have a partial block
    unsigned partial_blk_k = gemm_args_.dims.a.n - (n_blk_k_blocks * blk_k);
    int partial_block = !!(partial_blk_k);

    // Wait for A, B to reside in scratch memory
    member.team_barrier();

    // Each thread calculates a single dot product in chunks of size blk_k
#pragma unroll
    for (int k = 1; k < n_blk_k_blocks + partial_block; ++k) {
      auto k_block_offset = k * blk_k;

      // Get this threads next blk_k entries from global memory
      // Each thread has its own copy of prefetch_reg_b. TeamThreadRange runs over all threads in the team.
      // TODO: only fetch partial_blk_k since the last fetch is out of bounds?
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_n / REG_N), [&](const int &thread_id) {
          auto thread_offset = thread_id + start_n;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
              for (int i = 0; i < REG_N; ++i)
                prefetch_reg_b[i] = svB(vlane_id + k_block_offset, thread_offset + i * STRIDE_N);
            });
        });

      // Get this threads next blk_k entries from global memory
      // Each thread has its own copy of prefetch_reg_b. TeamThreadRange runs over all threads in the team.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
          auto thread_offset = thread_id + start_m;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
              for (int i = 0; i < REG_M; ++i)
                prefetch_reg_a[i] = svA(thread_offset + i * STRIDE_M, vlane_id + k_block_offset);
            });
        });

      // Multiply
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
	  Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_n / REG_N), [&](const int &vlane_id) {
#pragma unroll
              for (int k = 0; k < blk_k; ++k) {
#pragma unroll
                for (int m = 0; m < REG_M; ++m) {
                  reg_a[m] = svA_scr(k, thread_id + m * STRIDE_M);
                }

#pragma unroll
                for (int n = 0; n < REG_N; ++n) {
                  reg_b[n] = svB_scr(k, vlane_id + n * STRIDE_N);
                }

#pragma unroll
                for (int m = 0; m < REG_M; ++m) {
#pragma unroll
                  for (int n = 0; n < REG_N; ++n) {
                    reg_c[m][n] += reg_a[m] * reg_b[n] * gemm_args_.alpha;
                  }
                }
              }
	    });
	});

      // Wait for:
      //   1. prefetch_regs to be populated
      //   2. for shmem to no longer be read from
      member.team_barrier();

      // populate shmem from prefetch registers. Each thread has its own copy of prefetch_reg_a.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_n / REG_N), [&](const int &thread_id) {
          auto thread_offset = thread_id;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
              for (int i = 0; i < REG_N; ++i) {
                svB_scr(vlane_id, thread_offset + i * STRIDE_N) = prefetch_reg_b[i];
              }
            });
        });

      // populate shmem from prefetch registers. Each thread has its own copy of prefetch_reg_b.
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
          auto thread_offset = thread_id;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_k), [&](const int &vlane_id) {
#pragma unroll
              for (int i = 0; i < REG_M; ++i)
                svA_scr(vlane_id, thread_offset + i * STRIDE_M) = prefetch_reg_a[i];
            });
        });

      // Wait for shmem stores to land before performing next blk_k multiply
      member.team_barrier();

    } // end n_blk_k_blocks loop

    // Multiply last block, may be a partial block
    partial_blk_k = partial_block ? partial_blk_k : blk_k;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_n / REG_N), [&](const int &vlane_id) {
#pragma unroll
            for (int k = 0; k < partial_blk_k; ++k) {
#pragma unroll
              for (int m = 0; m < REG_M; ++m) {
                reg_a[m] = svA_scr(k, thread_id + m * STRIDE_M);
              }

#pragma unroll
              for (int n = 0; n < REG_N; ++n) {
                reg_b[n] = svB_scr(k, vlane_id + n * STRIDE_N);
              }

#pragma unroll
              for (int m = 0; m < REG_M; ++m) {
#pragma unroll
                for (int n = 0; n < REG_N; ++n) {
                  reg_c[m][n] += reg_a[m] * reg_b[n] * gemm_args_.alpha;
                }
              }
            }
          });
      });
    } // release register and shmem allocations

    shmem_size = max(svC_dims, shmem_size); // blk_k x blk_m + blk_k x blk_n
    view_type_2d_scratch svC_scr(member.team_scratch(0), blk_m, blk_n);

    // store results back to global memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, blk_m / REG_M), [&](const int &thread_id) {
        auto thread_m_offset = thread_id + start_m;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, blk_n / REG_N), [&](const int &vlane_id) {
            auto thread_n_offset = vlane_id + start_n;
#pragma unroll
            for (int m = 0; m < REG_M; ++m) {
              int cm = thread_m_offset + m * STRIDE_M;
#pragma unroll
              for (int n = 0; n < REG_N; ++n) {
                int cn = thread_n_offset + n * STRIDE_N;
                svC_scr(cm, cn) = reg_c[m][n] + svC(cm, cn) * gemm_args_.beta;
              }
            }
          });
      });

    member.team_barrier();

    // store results back to global memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svC.extent(0)), [&](const int &thread_id) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svC.extent(1)), [&](const int &vlane_id) {
            svC(thread_id, vlane_id) = svC_scr(thread_id, vlane_id);
          });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3Tag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svB = Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), i);
    auto svC = Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), i);

    view_type_2d_scratch svA_scr(member.team_scratch(0), svA.extent(0),  svA.extent(1));
    view_type_2d_scratch svB_scr(member.team_scratch(0), svB.extent(0),  svB.extent(1));
    //view_type_2d_scratch svC_scr(member.team_scratch(0), svC.extent(0),  svC.extent(1));
    // 256 threads / team -- 8 teams / SM
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svB.extent(0)), [&](const int &i) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svB.extent(1)), [&](const int &j) {
        svB_scr(i,j) = svB(i,j); // TODO: reduce bank conflicts, if svB_scr LayoutLeft, writes would be 256 bytes apart
      });
    });

    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svA.extent(0)), [&](const int &i) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svA.extent(1)), [&](const int &j) {
        svA_scr(i,j) = svA(i,j); // TODO: reduce bank conflicts
      });
    });

    // Wait for A, B, C to reside in scratch memory
    member.team_barrier();

    // TODO: use 16x16 tiles
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, svC.extent(0)),[&](const int &row_idx) { // thread.x
      auto svA_row = Kokkos::subview(svA_scr, row_idx, Kokkos::ALL());
      // svA_row_scr = svA_row
      // member.team_barrier();
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 0, svC.extent(1)),[&](const int &col_idx) { // thread.y
        auto svB_col = Kokkos::subview(svB_scr, Kokkos::ALL(), col_idx); // 256 bytes apart
        auto svC_ele = Kokkos::subview(svC, row_idx, col_idx);

        // TODO: Serial dot -- try to use some registers and then do a
        // multi-word write to C
        KokkosBatched::SerialGemm<Trans::Transpose, TransBType,
            BlockingType>::invoke(gemm_args_.alpha,
                                  svA_row,
                                  svB_col,
                                  gemm_args_.beta,
                                  svC_ele);
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamTagOptDivisor &, const MemberType &member) const {
    auto i = divisor_ * member.league_rank();
    for (size_t j = 0; j < divisor_; j++) {
      auto svA =
          Kokkos::subview(gemm_args_.A, i + j, Kokkos::ALL(), Kokkos::ALL());
      auto svB =
          Kokkos::subview(gemm_args_.B, i + j, Kokkos::ALL(), Kokkos::ALL());
      auto svC =
          Kokkos::subview(gemm_args_.C, i + j, Kokkos::ALL(), Kokkos::ALL());

      KokkosBatched::TeamGemm<MemberType, TransAType, TransBType,
                              BlockingType>::invoke(member, gemm_args_.alpha,
                                                    svA, svB, gemm_args_.beta,
                                                    svC);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamBatchDim3TagOptDivisor &,
                  const MemberType &member) const {
    auto i = divisor_ * member.league_rank();
    for (size_t j = 0; j < divisor_; j++) {
      auto svA =
          Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), i + j);
      auto svB =
          Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), i + j);
      auto svC =
          Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), i + j);

      KokkosBatched::TeamGemm<MemberType, TransAType, TransBType,
                              BlockingType>::invoke(member, gemm_args_.alpha,
                                                    svA, svB, gemm_args_.beta,
                                                    svC);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamVectorTag &, const MemberType &member) const {
    auto team_idx = member.league_rank();
    auto svA =
        Kokkos::subview(gemm_args_.A, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svB =
        Kokkos::subview(gemm_args_.B, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svC =
        Kokkos::subview(gemm_args_.C, team_idx, Kokkos::ALL(), Kokkos::ALL());


    KokkosBatched::TeamVectorGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member,
                                                        gemm_args_.alpha, svA,
                                                        svB, gemm_args_.beta,
                                                        svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamVectorBatchDim3Tag &,
                  const MemberType &member) const {
    auto team_idx = member.league_rank();
    auto svA =
        Kokkos::subview(gemm_args_.A, Kokkos::ALL(), Kokkos::ALL(), team_idx);
    auto svB =
        Kokkos::subview(gemm_args_.B, Kokkos::ALL(), Kokkos::ALL(), team_idx);
    auto svC =
        Kokkos::subview(gemm_args_.C, Kokkos::ALL(), Kokkos::ALL(), team_idx);

    KokkosBatched::TeamVectorGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member,
                                                        gemm_args_.alpha, svA,
                                                        svB, gemm_args_.beta,
                                                        svC);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamSimdTag &, const MemberType &member) const {
    auto i = member.league_rank();
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, gemm_args_.Cv.ivec_4d.extent(3)),
        [&](const int &vector_lane) {
          auto svA = Kokkos::subview(gemm_args_.Av.ivec_4d, i, Kokkos::ALL(),
                                     Kokkos::ALL(), vector_lane);
          auto svB = Kokkos::subview(gemm_args_.Bv.ivec_4d, i, Kokkos::ALL(),
                                     Kokkos::ALL(), vector_lane);
          auto svC = Kokkos::subview(gemm_args_.Cv.ivec_4d, i, Kokkos::ALL(),
                                     Kokkos::ALL(), vector_lane);

          KokkosBatched::Gemm<MemberType, TransAType, TransBType, AlgoMode,
                              BlockingType>::invoke(member, gemm_args_.alpha,
                                                    svA, svB, gemm_args_.beta,
                                                    svC);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamSimdBatchDim4Tag &,
                  const MemberType &member) const {
    auto i = member.league_rank();
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, gemm_args_.Cv.ivec_4d.extent(0)),
        [&](const int &vector_lane) {
          auto svA = Kokkos::subview(gemm_args_.Av.ivec_4d, vector_lane,
                                     Kokkos::ALL(), Kokkos::ALL(), i);
          auto svB = Kokkos::subview(gemm_args_.Bv.ivec_4d, vector_lane,
                                     Kokkos::ALL(), Kokkos::ALL(), i);
          auto svC = Kokkos::subview(gemm_args_.Cv.ivec_4d, vector_lane,
                                     Kokkos::ALL(), Kokkos::ALL(), i);

          KokkosBatched::Gemm<MemberType, TransAType, TransBType, AlgoMode,
                              BlockingType>::invoke(member, gemm_args_.alpha,
                                                    svA, svB, gemm_args_.beta,
                                                    svC);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialSimdTag &, const MemberType &member) const {
    Kokkos::abort("SerialSimdTag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialSimdBatchDim3Tag &,
                  const MemberType &member) const {
    Kokkos::abort("SerialSimdBatchDim3Tag not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt1 &, const MemberType &member) const {
    Kokkos::abort("SerialTagOpt1 not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt1 &,
                  const MemberType &member) const {
    Kokkos::abort("SerialBatchDim3TagOpt1 not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialTagOpt2Tiled &, const MemberType &member) const {
    Kokkos::abort("SerialTagOpt2Tiled not supported using RangePolicy.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const SerialBatchDim3TagOpt2Tiled &,
                  const MemberType &member) const {
    Kokkos::abort(
        "SerialBatchDim3TagOpt2Tiled not supported using RangePolicy.");
  }
};

template <class TransAType, class TransBType, class BlockingType, class AlgoTag,
          class device_type>
void __do_gemm_parallel_batched_template_range_policy(options_t options,
                                                      gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::RangePolicy<AlgoTag, execution_space>;
  using functor_type =
      parallel_batched_gemm_range_policy<TransAType, TransBType, BlockingType>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto batch_size    = options.start.c.k;
  size_t divisor     = 1;
  Kokkos::Timer timer;

  STATUS;

  if (std::is_same<AlgoTag, SerialTagOpt1>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt1>::value) {
    // NOTE: Intentionally leave AlgoTag at Opt1 on host for perf test
    // NOTE: Intentionally stay at Opt1 even if batch_size >=
    // backend_thread_threshold
    divisor = gemm_args.dims.c.n;
    batch_size *= divisor;
  }

  if (std::is_same<AlgoTag, SerialTagOpt2>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt2>::value ||
      std::is_same<AlgoTag, SerialTagOpt2Tiled>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt2Tiled>::value) {
    // NOTE: Intentionally leave AlgoTag at Opt1 on host for perf test
    // NOTE: Intentionally stay at Opt1 even if batch_size >=
    // backend_thread_threshold
    divisor = gemm_args.dims.c.n * gemm_args.dims.c.m;
    batch_size *= divisor;
  }

  functor_type parallel_batched_gemm_functor(
      gemm_args, options.blas_args.batch_size_last_dim, divisor, options.tile.m,
      options.tile.n);

  if (std::is_same<AlgoTag, SerialSimdTag>::value ||
      std::is_same<AlgoTag, SerialSimdBatchDim3Tag>::value) {
    batch_size = options.blas_args.batch_size_last_dim
                     ? gemm_args.Cv.vec_3d.extent(2)
                     : gemm_args.Cv.vec_3d.extent(0);
  }

  for (uint32_t i = 0; i < warm_up_n; i++) {
    Kokkos::parallel_for("parallelBatchedWarmUpLoopGemm",
                         policy_type(0, batch_size),
                         parallel_batched_gemm_functor);
    Kokkos::fence();
  }

  timer.reset();
  for (uint32_t i = 0; i < n; i++) {
    Kokkos::parallel_for("parallelBatchedTimedLoopGemm",
                         policy_type(0, batch_size),
                         parallel_batched_gemm_functor);
    Kokkos::fence();
  }

  __gemm_output_csv_row(options, gemm_args, timer.seconds());

  return;
}

template <class TransAType, class TransBType, class BlockingType, class AlgoTag,
          class device_type, int reg_m = 1, int reg_n = 1, int tile_n = 1, class algo_mode = void>
void __do_gemm_parallel_batched_template(options_t options,
                                         gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::TeamPolicy<AlgoTag, execution_space>;
  using member_type     = typename policy_type::member_type;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto divisor       = gemm_args.bp.divisor;
  auto league_size   = options.start.c.k / divisor;
  auto team_size     = gemm_args.bp.team_size;
  auto vector_len    = gemm_args.bp.vector_len;
  Kokkos::Timer timer;

  if (std::is_same<AlgoTag, SerialTag>::value ||
      std::is_same<AlgoTag, SerialBatchDim3Tag>::value ||
      std::is_same<AlgoTag, SerialTagOpt1>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt1>::value ||
      std::is_same<AlgoTag, SerialTagOpt2>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt2>::value ||
      std::is_same<AlgoTag, SerialTagOpt2Tiled>::value ||
      std::is_same<AlgoTag, SerialBatchDim3TagOpt2Tiled>::value ||
      std::is_same<AlgoTag, SerialSimdTag>::value ||
      std::is_same<AlgoTag, SerialSimdBatchDim3Tag>::value) {
    return __do_gemm_parallel_batched_template_range_policy<
        TransAType, TransBType, BlockingType, AlgoTag, device_type>(options,
                                                                    gemm_args);
  }

  if (std::is_same<AlgoTag, TeamSimdTag>::value ||
      std::is_same<AlgoTag, TeamSimdBatchDim4Tag>::value) {
    league_size = options.blas_args.batch_size_last_dim
                      ? gemm_args.Cv.ivec_4d.extent(3)
                      : gemm_args.Cv.ivec_4d.extent(0);
    vector_len  = simd_vector_size /
                 simd_internal_vector_size;  // TODO: use bp.vector_len?
  }

  if (std::is_same<AlgoTag, TeamTagOpt1>::value ||
      std::is_same<AlgoTag, TeamBatchDim3TagOpt1>::value) {
    // NOTE: Intentionally leave AlgoTag at Opt1 on host for perf test
    // NOTE: Intentionally stay at Opt1 even if batch_size >=
    // backend_thread_threshold
    divisor = gemm_args.dims.c.n;
    league_size *= divisor;
  }

  STATUS;

  constexpr int stride_n  = tile_n / reg_n;   // 32 / 4 = 8
  constexpr int tile_m    = stride_n * reg_m; // 8 * 4  = 32
  constexpr int stride_m  = tile_m / reg_m;   // 32 / 4 = 8
  constexpr int tile_k    = stride_m;         // = 8

  using functor_type =
      parallel_batched_gemm<member_type, TransAType, TransBType, BlockingType,
                            reg_m, reg_n, stride_m, stride_n,
                            algo_mode>;

  functor_type parallel_batched_gemm_functor(gemm_args, options.blas_args.batch_size_last_dim, divisor, tile_m, tile_n, tile_k);



  // For pre-fetch version:
  size_t shmem_size =
      view_type_2d_scratch::shmem_size(tile_m, tile_k) +
      view_type_2d_scratch::shmem_size(tile_k, tile_n);

  if (std::is_same<AlgoTag, TeamShmemTag>::value ||
      std::is_same<AlgoTag, TeamShmemBatchDim3Tag>::value) {
    shmem_size =
      view_type_2d_scratch::shmem_size(gemm_args.dims.a.m, gemm_args.dims.a.n) +
      view_type_2d_scratch::shmem_size(gemm_args.dims.b.m, gemm_args.dims.b.n);
  }
  // --For tiling without pre-fetch version--
  // size_t shmem_size =
  //     view_type_2d_scratch::shmem_size(options.tile.m, gemm_args.dims.a.n) +
  //     view_type_2d_scratch::shmem_size(gemm_args.dims.b.m, options.tile.n);

  if (std::is_same<AlgoTag, TeamTag>::value ||
      std::is_same<AlgoTag, TeamBatchDim3Tag>::value) {
    league_size *= parallel_batched_gemm_functor.n_sub_blocks;
    team_size = tile_m / reg_m;
    vector_len = tile_n / reg_n;
  }

  if (options.blas_args.use_auto) {
    for (uint32_t i = 0; i < warm_up_n; i++) {
      Kokkos::parallel_for("parallelBatchedWarmUpLoopGemm",
                           policy_type(league_size, Kokkos::AUTO, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
                           parallel_batched_gemm_functor);
      Kokkos::fence();
    }

    timer.reset();
    for (uint32_t i = 0; i < n; i++) {
      Kokkos::parallel_for("parallelBatchedTimedLoopGemm",
                           policy_type(league_size, Kokkos::AUTO, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
                           parallel_batched_gemm_functor);
      Kokkos::fence();
    }
  } else {
    for (uint32_t i = 0; i < warm_up_n; i++) {
      Kokkos::parallel_for("parallelBatchedWarmUpLoopGemm",
                           policy_type(league_size, team_size, vector_len).set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
                           parallel_batched_gemm_functor);
      Kokkos::fence();
    }

    timer.reset();
    for (uint32_t i = 0; i < n; i++) {
      Kokkos::parallel_for("parallelBatchedTimedLoopGemm",
                           policy_type(league_size, team_size, vector_len).set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
                           parallel_batched_gemm_functor);
      Kokkos::fence();
    }
  }

  __gemm_output_csv_row(options, gemm_args, timer.seconds());

  return;
}

template <class algo_tag, class blocking_type, class device_type,
          class algo_mode = void>
void __do_gemm_parallel_batched(options_t options, gemm_args_t gemm_args) {
  char a  = gemm_args.transA;
  char b  = gemm_args.transB;
  using N = Trans::NoTranspose;
  using T = Trans::Transpose;
  // using C = Trans::ConjTranspose;

  STATUS;

  if (a == 'N' && b == 'N') {
    if (gemm_args.dims.c.n >= 32 && gemm_args.dims.c.m >= 32) {
      __do_gemm_parallel_batched_template<N, N, blocking_type, algo_tag,
                                          device_type, 4, 4, 32, algo_mode>(options,
                                                                            gemm_args);
    } else {
      __do_gemm_parallel_batched_template<N, N, blocking_type, algo_tag,
                                          device_type, 1, 1, 1, algo_mode>(options,
                                                                           gemm_args);
    }
  } else if (a == 'N' && b == 'T') {
    __do_gemm_parallel_batched_template<N, T, blocking_type, algo_tag,
                                        device_type, 1, 1, 1, algo_mode>(options,
                                                                         gemm_args);
    //} else if (a == 'N' && b == 'C') {
    //  __do_gemm_parallel_batched_template<N, C, blocking_type, algo_tag,
    //  device_type>(options, gemm_args);
  } else if (a == 'T' && b == 'N') {
    __do_gemm_parallel_batched_template<T, N, blocking_type, algo_tag,
                                        device_type, 1, 1, 1, algo_mode>(options,
                                                                         gemm_args);
  } else if (a == 'T' && b == 'T') {
    __do_gemm_parallel_batched_template<T, T, blocking_type, algo_tag,
                                          device_type, 1, 1, 1, algo_mode>(options,
                                                                           gemm_args);
    //} else if (a == 'T' && b == 'C') {
    //  __do_gemm_parallel_batched_template<T, C, blocking_type, algo_tag,
    //  device_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'N') {
    //  __do_gemm_parallel_batched_template<C, N, blocking_type, algo_tag,
    //  device_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'T') {
    //  __do_gemm_parallel_batched_template<C, T, blocking_type, algo_tag,
    //  device_type>(options, gemm_args);
    //} else if (a == 'C' && b == 'C') {
    //  __do_gemm_parallel_batched_template<C, C, blocking_type, algo_tag,
    //  device_type>(options, gemm_args);
  } else {
    FATAL_ERROR("Bad gemm_args TransA or TransB value");
  }

  return;
}

template <class TransAType, class TransBType, class BlockingType>
struct parallel_batched_gemm_experiment1 {
  gemm_args_t gemm_args_;

  parallel_batched_gemm_experiment1(gemm_args_t gemm_args)
      : gemm_args_(gemm_args) {}

  KOKKOS_INLINE_FUNCTION

  void operator()(const SerialTag &, const int &i) const {
    auto svA = Kokkos::subview(gemm_args_.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(gemm_args_.B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(gemm_args_.C, i, Kokkos::ALL(), Kokkos::ALL());

    // Uses two serial for-loops internally
    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args_.alpha, svA, svB, gemm_args_.beta, svC);
  }
};

/**
 * 1. parallel_for(rangePolicy<Kokkos::DefaultExecutionSpace>(N)): serialGemm
 *
 */
template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment1(options_t options, gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::RangePolicy<SerialTag, execution_space>;
  using functor_type =
      parallel_batched_gemm_experiment1<TransAType, TransBType, BlockingType>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto k             = options.start.c.k;
  Kokkos::Timer timer;
  STATUS;

  functor_type experiment1_functor(gemm_args);

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment1Gemm",
                         policy_type(0, k), experiment1_functor);
  }
  Kokkos::fence();

  timer.reset();
  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment1Gemm",
                         policy_type(0, k), experiment1_functor);
  }
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment1");
  return;
}

template <class TransAType, class TransBType, class BlockingType,
          class MemberType>
struct parallel_batched_gemm_experiment2_3_4 {
  gemm_args_t gemm_args_;

  parallel_batched_gemm_experiment2_3_4(gemm_args_t gemm_args)
      : gemm_args_(gemm_args) {}

  // Experiment 2
  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamVectorTag &, const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(gemm_args_.A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(gemm_args_.B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(gemm_args_.C, i, Kokkos::ALL(), Kokkos::ALL());

    // Uses TeamThreadRange over C-rows
    //        ThreadVectorRange over C-cols
    KokkosBatched::TeamVectorGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member,
                                                        gemm_args_.alpha, svA,
                                                        svB, gemm_args_.beta,
                                                        svC);
  }

  // Experiment 3
  KOKKOS_INLINE_FUNCTION
  void operator()(const LayoutLeftTag &, const MemberType &member) const {
    auto team_idx = member.league_rank();
    auto svA =
        Kokkos::subview(gemm_args_.A, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svB =
        Kokkos::subview(gemm_args_.B, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svC =
        Kokkos::subview(gemm_args_.C, team_idx, Kokkos::ALL(), Kokkos::ALL());

    // TeamThreadRange:   splits the index range over the threads of the team
    // ThreadVectorRange: splits the index range over the vector lanes of the
    // calling thread

    auto svC_cols = svC.extent(1);
    // In a given team, for each vector lane, compute zero or more output
    // columns of C depending on the index range
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, svC_cols), [&](const int &lane_idx) {
          auto svB_col = Kokkos::subview(svB, Kokkos::ALL(), lane_idx);
          auto svC_col = Kokkos::subview(svC, Kokkos::ALL(), lane_idx);
          // TeamGemm Calls TeamThreadRange over M*N meaning the flat M*N array
          // is split over all threads of the team
          KokkosBatched::TeamGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member,
                                                        gemm_args_.alpha, svA,
                                                        svB_col,
                                                        gemm_args_.beta,
                                                        svC_col);
        });
  }

  // TODO: Why is this faster than the LayoutLeftTag operator above for both
  // LayoutLeft and LayoutRight? Experiment 4
  KOKKOS_INLINE_FUNCTION
  void operator()(const LayoutRightTag &, const MemberType &member) const {
    auto team_idx = member.league_rank();
    auto svA =
        Kokkos::subview(gemm_args_.A, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svB =
        Kokkos::subview(gemm_args_.B, team_idx, Kokkos::ALL(), Kokkos::ALL());
    auto svC =
        Kokkos::subview(gemm_args_.C, team_idx, Kokkos::ALL(), Kokkos::ALL());

    // TeamThreadRange:   splits the index range over the threads of the team
    // ThreadVectorRange: splits the index range over the vector lanes of the
    // calling thread

    auto svC_rows = svC.extent(0);
    // In a given team, for each vector lane, compute zero or more output rows
    // of C depending on the index range
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(member, svC_rows), [&](const int &lane_idx) {
          auto svA_row = Kokkos::subview(svA, lane_idx, Kokkos::ALL());
          auto svC_row = Kokkos::subview(svC, lane_idx, Kokkos::ALL());
          // TeamGemm Calls TeamThreadRange over M*N meaning the flat M*N array
          // is split over all threads of the team
          KokkosBatched::TeamGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member,
                                                        gemm_args_.alpha,
                                                        svA_row, svB,
                                                        gemm_args_.beta,
                                                        svC_row);
        });
  }
};

/**
 * 2. case a)
 * parallel_for(teamPolicy): TeamVectorGemm
 *
 */
template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment2(options_t options, gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::TeamPolicy<TeamVectorTag, execution_space>;
  using member_type     = typename policy_type::member_type;
  using functor_type =
      parallel_batched_gemm_experiment2_3_4<TransAType, TransBType,
                                            BlockingType, member_type>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto league_size   = options.start.c.k;
  Kokkos::Timer timer;
  STATUS;

  functor_type experiment2_functor(gemm_args);

  auto team_size  = gemm_args.bp.team_size;
  auto vector_len = gemm_args.bp.vector_len;

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment2Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment2_functor);
  }
  Kokkos::fence();

  timer.reset();

  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment2Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment2_functor);
  }
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment2");
  return;
}

/**
 * 3. case b)
 *    parallel_for(teamPolicy):
 *      parallel_for(TeamThreadRange):
 *         VectorGemm
 *
 * VectorGemm has not been implemented!
 * I think this experiment can be removed. TeamGemm calls TeamThreadRange
 * internally! TeamVectorGemm calls both TeamThreadRange and ThreadVectorRange
 * internally!
 */
template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment3(options_t options, gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  // using layout_tag = std::conditional<std::is_same<default_layout,
  // Kokkos::LayoutLeft>::value, LayoutLeftTag, LayoutRightTag>::type;
  using policy_type = Kokkos::TeamPolicy<LayoutLeftTag, execution_space>;
  using member_type = typename policy_type::member_type;
  using functor_type =
      parallel_batched_gemm_experiment2_3_4<TransAType, TransBType,
                                            BlockingType, member_type>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto league_size   = options.start.c.k;
  Kokkos::Timer timer;
  STATUS;

  functor_type experiment3_functor(gemm_args);

  auto team_size  = gemm_args.bp.team_size;
  auto vector_len = gemm_args.bp.vector_len;

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment3Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment3_functor);
  }
  Kokkos::fence();

  timer.reset();

  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment3Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment3_functor);
  }
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment3");
  return;
}

/**
 * 4. case c)
 * parallel_for(teamPolicy):
 *      parallel_for(ThreadVectorRange)
 *        TeamGemm
 */
template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment4(options_t options, gemm_args_t gemm_args) {
  using execution_space = typename device_type::execution_space;
  // using layout_tag = std::conditional<std::is_same<default_layout,
  // Kokkos::LayoutLeft>::value, LayoutLeftTag, LayoutRightTag>::type;
  using policy_type = Kokkos::TeamPolicy<LayoutRightTag, execution_space>;
  using member_type = typename policy_type::member_type;
  using functor_type =
      parallel_batched_gemm_experiment2_3_4<TransAType, TransBType,
                                            BlockingType, member_type>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto league_size   = options.start.c.k;
  Kokkos::Timer timer;
  STATUS;

  functor_type experiment4_functor(gemm_args);

  auto team_size  = gemm_args.bp.team_size;
  auto vector_len = gemm_args.bp.vector_len;

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment4Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment4_functor);
  }
  Kokkos::fence();

  timer.reset();

  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment4Gemm",
                         policy_type(league_size, team_size, vector_len),
                         experiment4_functor);
  }
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment4");
  return;
}

template <class SimdViewType, class TransAType, class TransBType,
          class BlockingType>
class parallel_batched_gemm_experiment5 {
 private:
  SimdViewType &A, &B, &C;
  gemm_args_t gemm_args;

 public:
  parallel_batched_gemm_experiment5(SimdViewType &_A, SimdViewType &_B,
                                    SimdViewType &_C, gemm_args_t _gemm_args)
      : A(_A), B(_B), C(_C), gemm_args(_gemm_args) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const SimdCpuTag &, const int &i) const {
    auto svA = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

    // Uses two serial for-loops internally
    KokkosBatched::SerialGemm<TransAType, TransBType, BlockingType>::invoke(
        gemm_args.alpha, svA, svB, gemm_args.beta, svC);
  }
};

/**
 * 5.
 * parallel_for(RangePolicy<Kokkos:DefaultHostExecutionSpace>(N/vl+(N%vl>0)>):
 * serialGemm
 *
 * Not portable to GPU
 */
template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment5(options_t options, gemm_args_t gemm_args) {
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::RangePolicy<SimdCpuTag, execution_space>;

  // Construct the SimdType
  using scalar_type = typename view_type_3d::value_type;
  constexpr int vl =
      KokkosBatched::DefaultVectorLength<scalar_type, execution_space>::value;
  using simd_type =
      KokkosBatched::Vector<KokkosBatched::SIMD<scalar_type>, simd_vector_size>;
  using simd_view_type =
      Kokkos::View<simd_type ***, default_layout, default_device>;
  using functor_type =
      parallel_batched_gemm_experiment5<simd_view_type, TransAType, TransBType,
                                        BlockingType>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto k             = options.start.c.k;
  Kokkos::Timer timer;
  auto simd_batch_size = k / vl + (k % vl > 0);
  STATUS;

  // Increases each array size by sizeof(scalar_type) * (vl-1) bytes!
  simd_view_type A("A", simd_batch_size, gemm_args.A.extent(0),
                   gemm_args.A.extent(1));
  simd_view_type B("B", simd_batch_size, gemm_args.B.extent(0),
                   gemm_args.B.extent(1));
  simd_view_type C("C", simd_batch_size, gemm_args.C.extent(0),
                   gemm_args.C.extent(1));

  // uint64_t seed = Kokkos::Impl::clock_tic();
  // Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
  // Kokkos::fill_random(A, rand_pool,
  // Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
  // simd_type>::max()); Kokkos::fill_random(B, rand_pool,
  // Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
  // simd_type>::max()); Kokkos::fill_random(C, rand_pool,
  // Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
  // simd_type>::max()); execution_space::fence();

  functor_type experiment5_functor(A, B, C, gemm_args);

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment5Gemm",
                         policy_type(0, simd_batch_size), experiment5_functor);
  }
  Kokkos::fence();

  timer.reset();

  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment5Gemm",
                         policy_type(0, simd_batch_size), experiment5_functor);
  }
  Kokkos::fence();

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment5");
#else
  std::cerr
      << std::string(__func__)
      << " disabled since KOKKOS_ENABLE_CUDA or KOKKOS_ENABLE_HIP is defined."
      << std::endl;
#endif  // !KOKKOS_ENABLE_CUDA || !KOKKOS_ENABLE_HIP
  return;
}

template <class MemberType, class SimdViewType, class TransAType,
          class TransBType, class BlockingType>
class parallel_batched_gemm_experiment6 {
 private:
  SimdViewType &A, &B, &C;
  gemm_args_t gemm_args;

 public:
  parallel_batched_gemm_experiment6(SimdViewType &_A, SimdViewType &_B,
                                    SimdViewType &_C, gemm_args_t _gemm_args)
      : A(_A), B(_B), C(_C), gemm_args(_gemm_args) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &member) const {
    auto i   = member.league_rank();
    auto svA = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
    auto svB = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
    auto svC = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());

    // Uses two serial for-loops internally
    KokkosBatched::TeamVectorGemm<MemberType, TransAType, TransBType,
                                  BlockingType>::invoke(member, gemm_args.alpha,
                                                        svA, svB,
                                                        gemm_args.beta, svC);
  }
};

template <class TransAType, class TransBType, class BlockingType,
          class device_type>
void __do_gemm_parallel_experiment6(options_t options, gemm_args_t gemm_args) {
#if 0
  using execution_space = typename device_type::execution_space;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  // Construct the vector type
  using scalar_type = typename view_type_3d::value_type;
  constexpr int vl =
      KokkosBatched::DefaultVectorLength<scalar_type, execution_space>::value;
  constexpr int il =
      KokkosBatched::DefaultInternalVectorLength<scalar_type, execution_space>::value;
  using view_type = Kokkos::View<scalar_type***[vl], default_layout, default_device>;
  using vector_view_type = Kokkos::View<vector_type***, default_layout, default_device>;
  using internal_vector_view_type = Kokkos::View<internal_vector_type***, default_layout, default_device>;
  using functor_type =
      parallel_batched_gemm_experiment6<member_type, internal_vector_view_type,
                                        TransAType, TransBType, BlockingType>;

  uint32_t warm_up_n = options.warm_up_n;
  uint32_t n         = options.n;
  auto k             = options.start.c.k;
  Kokkos::Timer timer;
  auto simd_batch_size = k / vl + (k % vl > 0);
  STATUS;

  // Construct matrices
  vector_view_type A_vector("A_vector", simd_batch_size, gemm_args.A.extent(0), gemm_args.A.extent(1));
  view_type A((scalar_type *)A_vector.data(), simd_batch_size, gemm_args.A.extent(0), gemm_args.A.extent(1));
  internal_vector_view_type A_vector_internal(A_vector.data(), simd_batch_size, gemm_args.A.extent(0), gemm_args.A.extent(1));

  vector_view_type B_vector("B_vector", simd_batch_size, gemm_args.B.extent(0), gemm_args.B.extent(1));
  view_type B((scalar_type *)B_vector.data(), simd_batch_size, gemm_args.B.extent(0), gemm_args.B.extent(1));
  internal_vector_view_type B_vector_internal(B_vector.data(), simd_batch_size, gemm_args.B.extent(0), gemm_args.B.extent(1));

  vector_view_type C_vector("C_vector", simd_batch_size, gemm_args.C.extent(0), gemm_args.C.extent(1));
  view_type C((scalar_type *)C_vector.data(), simd_batch_size, gemm_args.C.extent(0), gemm_args.C.extent(1));
  internal_vector_view_type C_vector_internal(C_vector.data(), simd_batch_size, gemm_args.C.extent(0), gemm_args.C.extent(1));

  uint64_t seed = Kokkos::Impl::clock_tic();
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
  Kokkos::fill_random(A, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, scalar_type>::max());
  Kokkos::fill_random(B, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, scalar_type>::max());
  Kokkos::fill_random(C, rand_pool, Kokkos::rand<Kokkos::Random_XorShift64<execution_space>, scalar_type>::max());
  Kokkos::fence();

  functor_type experiment6_functor(A_vector_internal, B_vector_internal, C_vector_internal, gemm_args);

  for (uint32_t i = 0; i < warm_up_n; ++i) {
    Kokkos::parallel_for("parallelBatchedUntimedExperiment6Gemm",
                         policy_type(simd_batch_size, Kokkos::AUTO, vl/il), experiment6_functor);
    Kokkos::fence();
  }

  timer.reset();
  for (uint32_t i = 0; i < n; ++i) {
    Kokkos::parallel_for("parallelBatchedTimedExperiment6Gemm",
                         policy_type(simd_batch_size, Kokkos::AUTO, vl/il), experiment6_functor);
    Kokkos::fence();
  }

  __gemm_output_csv_row(options, gemm_args, timer.seconds(), "experiment6");
#endif
  return;
}

/**
 * Check difference of scalars expected and actual at indexes i,j,k
 * @var expected: The expected result.
 * @var actual:   The actual result.
 * @var epsilon:  The tolerance to use when comparing.
 * @return true if the comparison fails and false if the comparison succeeds.
 */
template <class ViewType>
static inline bool __gemm_print_compare_failure(ViewType h_expected,
                                                ViewType h_actual, int i, int j,
                                                int k, double epsilon) {
  STATUS;
  auto diff = static_cast<double>(Kokkos::Experimental::fabs(
      static_cast<double>(h_expected(i, j, k) - h_actual(i, j, k))));

  if (diff > epsilon) {
    printf(
        "fabs(expected(%d,%d,%d):%g - actual(%d,%d,%d):%g):%g > epsilon:%g\n",
        i, j, k, static_cast<double>(h_expected(i, j, k)), i, j, k,
        static_cast<double>(h_actual(i, j, k)), diff, epsilon);
    FATAL_ERROR("Comparison failure!");
    return true;
  }
  return false;
}

/**
 * Compare all values of expected with all values of actual.
 * @var expected: the expected results
 * @var actual:   the actual results
 * @return false if expected matches actual within epsilon, otherwise true.
 */
template <class ScalarType, class LayoutType>
static inline bool __gemm_do_compare(view_type_3d expected,
                                     view_type_3d actual) {
  double epsilon = Test::epsilon<ScalarType>::value * 1e3;
  STATUS;

  typename view_type_3d::HostMirror h_expected =
      Kokkos::create_mirror_view(expected);
  typename view_type_3d::HostMirror h_actual =
      Kokkos::create_mirror_view(actual);

  // Copy to host for comparision
  Kokkos::deep_copy(h_expected, expected);
  Kokkos::deep_copy(h_actual, actual);
  Kokkos::fence();

  if (std::is_same<LayoutType, Kokkos::LayoutRight>::value) {
    for (size_t i = 0; i < h_expected.extent(0); i++) {
      for (size_t j = 0; j < h_expected.extent(1); j++) {
        for (size_t k = 0; k < h_expected.extent(2); k++) {
          if (__gemm_print_compare_failure<decltype(h_expected)>(
                  h_expected, h_actual, i, j, k, epsilon))
            return true;
        }
      }
    }
  }

  if (std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
    for (size_t k = 0; k < h_expected.extent(2); k++) {
      for (size_t j = 0; j < h_expected.extent(1); j++) {
        for (size_t i = 0; i < h_expected.extent(0); i++) {
          if (__gemm_print_compare_failure<decltype(h_expected)>(
                  h_expected, h_actual, i, j, k, epsilon))
            return true;
        }
      }
    }
  }

  return false;
}

template <class dstViewType>
static inline void __gemm_copy_simd_view_to_3d_view(gemm_simd_args_t src,
                                                    dstViewType dst,
                                                    options_t options) {
  using dst_scalar_type = typename dstViewType::value_type;
  using src_scalar_type = typename view_type_5d::value_type;
  size_t remainder, vector_batch_size, simd_batch_size, last_batch;
  bool data_layout_same_as_3d_view       = false;
  typename dstViewType::HostMirror h_dst = Kokkos::create_mirror_view(dst);
  typename view_type_4d::HostMirror h_src =
      Kokkos::create_mirror_view(src.mat_4d);
  Kokkos::deep_copy(h_src, src.mat_4d);
  Kokkos::fence();

  if (options.blas_args.batch_size_last_dim) {
    remainder         = dst.extent(2) % simd_internal_vector_size;
    vector_batch_size = src.ivec_4d.extent(0);
    simd_batch_size   = src.ivec_4d.extent(3);
    last_batch        = dst.extent(2);
    if (std::is_same<default_layout, Kokkos::LayoutRight>::value &&
        remainder == 0 && false)
      data_layout_same_as_3d_view = true;

  } else {
    remainder         = dst.extent(0) % simd_internal_vector_size;
    vector_batch_size = src.ivec_4d.extent(3);
    simd_batch_size   = src.ivec_4d.extent(0);
    last_batch        = dst.extent(0);
    if (std::is_same<default_layout, Kokkos::LayoutLeft>::value &&
        remainder == 0)
      data_layout_same_as_3d_view = true;
  }

  // When the batch_size is a multiple of the simd_vector_size and the
  // batch_size dimension is nearest to the simd_vector_size dimension, each
  // 2-rank matrix lies in the correct location and the data can simply be cast
  // to the 3d view.
  if (data_layout_same_as_3d_view) {
    // We can just re-cast the data to the 3d view but we'll copy it for
    // verification
    memcpy(h_dst.data(), h_src.data(),
           sizeof(dst_scalar_type) * dst.extent(0) * dst.extent(1) *
               dst.extent(2));
    Kokkos::deep_copy(dst, h_dst);
    Kokkos::fence();
    return;
  }

  // If the remainder is 0, we have simd_vector_size sub-batches to copy out...
  // this is a bad data access pattern but for these perf_tests we will support
  // it. If the remainder is non-zero, we have simd_vector_size sub-batches +
  // remainder to copy out.
  remainder += simd_internal_vector_size;

  // Views needed for slow manual copy
  using h_view_type_5d =
      Kokkos::View<src_scalar_type *****, default_layout, Kokkos::HostSpace>;
  using h_subview_type_2d =
      Kokkos::View<src_scalar_type **, Kokkos::LayoutStride, Kokkos::HostSpace>;
  using h_subview_type_3d =
      Kokkos::View<src_scalar_type ***, Kokkos::LayoutStride,
                   Kokkos::HostSpace>;
  using h_subview_type_4d =
      Kokkos::View<src_scalar_type ****, Kokkos::LayoutStride,
                   Kokkos::HostSpace>;
  h_view_type_5d h_src_raw;
  h_subview_type_4d h_sv0;
  h_subview_type_3d h_sv1;
  h_subview_type_2d h_sv2;

  // TODO: Clean everything below this point up...
  if (std::is_same<default_layout, Kokkos::LayoutRight>::value)
    h_src_raw =
        h_view_type_5d((src_scalar_type *)h_src.data(), src.ivec_4d.extent(0),
                       src.ivec_4d.extent(1), src.ivec_4d.extent(2),
                       src.ivec_4d.extent(3), simd_internal_vector_size);
  else
    h_src_raw = h_view_type_5d((src_scalar_type *)h_src.data(),
                               simd_internal_vector_size, src.ivec_4d.extent(0),
                               src.ivec_4d.extent(1), src.ivec_4d.extent(2),
                               src.ivec_4d.extent(3));

  // The below loops copies each corresponding 2-rank matrix within the simd
  // view back to the 3-rank view.
  for (size_t simd_internal_vec_idx = 0; simd_internal_vec_idx < remainder;
       simd_internal_vec_idx++) {
    if (std::is_same<default_layout, Kokkos::LayoutRight>::value)
      h_sv0 =
          Kokkos::subview(h_src_raw, Kokkos::ALL(), Kokkos::ALL(),
                          Kokkos::ALL(), Kokkos::ALL(), simd_internal_vec_idx);
    else
      h_sv0 = Kokkos::subview(h_src_raw, simd_internal_vec_idx, Kokkos::ALL(),
                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

    for (size_t vector_batch_idx = 0; vector_batch_idx < vector_batch_size;
         vector_batch_idx++) {
      if (options.blas_args.batch_size_last_dim)
        h_sv1 = Kokkos::subview(h_sv0, vector_batch_idx, Kokkos::ALL(),
                                Kokkos::ALL(), Kokkos::ALL());
      else
        h_sv1 = Kokkos::subview(h_sv0, Kokkos::ALL(), Kokkos::ALL(),
                                Kokkos::ALL(), vector_batch_idx);
      for (size_t simd_batch_size_idx = 0;
           simd_batch_size_idx < simd_batch_size; simd_batch_size_idx++) {
        if (options.blas_args.batch_size_last_dim)
          h_sv2 = Kokkos::subview(h_sv1, Kokkos::ALL(), Kokkos::ALL(),
                                  simd_batch_size_idx);
        else
          h_sv2 = Kokkos::subview(h_sv1, simd_batch_size_idx, Kokkos::ALL(),
                                  Kokkos::ALL());
        for (size_t m = 0; m < src.ivec_4d.extent(1); m++) {
          for (size_t n = 0; n < src.ivec_4d.extent(2); n++) {
            if (options.blas_args.batch_size_last_dim)
              h_dst(m, n,
                    simd_internal_vec_idx + simd_batch_size_idx +
                        vector_batch_idx) = h_sv2(m, n);
            else
              h_dst(simd_internal_vec_idx + simd_batch_size_idx +
                        vector_batch_idx,
                    m, n) = h_sv2(m, n);
          }
        }
        if (simd_internal_vec_idx + simd_batch_size_idx + vector_batch_idx ==
            last_batch - 1)
          goto out;
      }
    }
  }
out:
  Kokkos::deep_copy(dst, h_dst);
  Kokkos::fence();
}

/**
 * Compare all values of expected with all values of actual.
 * @var expected: the expected results
 * @var actual:   the actual results
 * @return false if expected matches actual within epsilon, otherwise true.
 */
template <class ScalarType, class LayoutType>
static inline bool __gemm_do_compare(view_type_3d expected,
                                     gemm_simd_args_t actual,
                                     options_t options) {
  decltype(expected) actual_data("actual_data", expected.extent(0),
                                 expected.extent(1), expected.extent(2));

  STATUS;

  // Copy the simd view to a 3d view for comparision.
  // NOTE: The raw results are different when batch_size % simd_vector_size !=
  // 0. Also note that when batch_size % simd_vector_size != 0, the simd
  // operation calculates results that we do not require. So, we end up running
  // an extra batch_size % simd_vector_size GEMMs!
  __gemm_copy_simd_view_to_3d_view(actual, actual_data, options);
  return __gemm_do_compare<ScalarType, LayoutType>(expected, actual_data);
}

template <class ScalarType, class LayoutType, class DeviceType>
static inline void __gemm_do_verify(options_t options, gemm_args_t gemm_args,
                                    void (*fn)(options_t, gemm_args_t)) {
  using execution_space = typename DeviceType::execution_space;
  // Just create "expected" types using non-simd types.
  decltype(gemm_args.C) C_expected;
  decltype(gemm_args.A) A_expected;
  decltype(gemm_args.B) B_expected;
  STATUS;

  if (options.blas_args.batch_size_last_dim) {
    C_expected = decltype(C_expected)("C_expected", gemm_args.dims.c.m,
                                      gemm_args.dims.c.n, gemm_args.dims.c.k);
    A_expected = decltype(A_expected)("A_expected", gemm_args.dims.a.m,
                                      gemm_args.dims.a.n, gemm_args.dims.a.k);
    B_expected = decltype(B_expected)("B_expected", gemm_args.dims.b.m,
                                      gemm_args.dims.b.n, gemm_args.dims.b.k);
  } else {
    C_expected = decltype(C_expected)("C_expected", gemm_args.dims.c.k,
                                      gemm_args.dims.c.m, gemm_args.dims.c.n);
    A_expected = decltype(A_expected)("A_expected", gemm_args.dims.a.k,
                                      gemm_args.dims.a.m, gemm_args.dims.a.n);
    B_expected = decltype(B_expected)("B_expected", gemm_args.dims.b.k,
                                      gemm_args.dims.b.m, gemm_args.dims.b.n);
  }

  // Initialize "expected" matrices.
  if (gemm_args.C.data() != nullptr) {
    Kokkos::deep_copy(C_expected, gemm_args.C);
    Kokkos::deep_copy(A_expected, gemm_args.A);
    Kokkos::deep_copy(B_expected, gemm_args.B);

    //printf("%lux%lux%lu", C_expected.extent(0), C_expected.extent(1), C_expected.extent(2));
    //exit(1);
    Kokkos::fence();  // Ensure that deep_copy has completed

    // Check that initial values match
    if (__gemm_do_compare<ScalarType, LayoutType>(C_expected, gemm_args.C))
      FATAL_ERROR("Inital values mismatch!");
  } else if (gemm_args.Cv.vec_3d.data() != nullptr) {
    __gemm_copy_simd_view_to_3d_view<decltype(C_expected)>(gemm_args.Cv,
                                                           C_expected, options);
    __gemm_copy_simd_view_to_3d_view<decltype(A_expected)>(gemm_args.Av,
                                                           A_expected, options);
    __gemm_copy_simd_view_to_3d_view<decltype(B_expected)>(gemm_args.Bv,
                                                           B_expected, options);

    // Check that initial values match
    if (__gemm_do_compare<ScalarType, LayoutType>(C_expected, gemm_args.Cv,
                                                  options))
      FATAL_ERROR("Inital values mismatch!");
  } else {
    FATAL_ERROR("Input arguments are empty!");
  }

  // Populate "expected" matrices via VanillaGemm
  Test::Functor_BatchedVanillaGEMM<decltype(A_expected), decltype(B_expected),
                                   decltype(C_expected), execution_space>
      vgemm;
  vgemm.A_t = toupper(gemm_args.transA) == 'T';
  vgemm.B_t = toupper(gemm_args.transB) == 'T';
  vgemm.A_c = vgemm.B_c     = false;
  vgemm.batch_size_last_dim = options.blas_args.batch_size_last_dim;
  vgemm.A                   = A_expected;
  vgemm.B                   = B_expected;
  vgemm.C                   = C_expected;
  vgemm.alpha               = gemm_args.alpha;
  vgemm.beta                = gemm_args.beta;
  vgemm.run();  // Compute C_expected

  // Run routine with warm_up_n = 1 and n = 0.
  auto warm_up_n_bak = options.warm_up_n;
  options.warm_up_n  = 1;
  auto n_bak         = options.n;
  options.n          = 0;
  fn(options, gemm_args);

  Kokkos::fence();  // Redundant fence.

  // Check the result
  if (gemm_args.C.data() != nullptr) {
    if (__gemm_do_compare<ScalarType, LayoutType>(C_expected, gemm_args.C))
      FATAL_ERROR("Result value mismatch!");
  }

  if (gemm_args.Cv.vec_3d.data() != nullptr) {
    if (__gemm_do_compare<ScalarType, LayoutType>(C_expected, gemm_args.Cv,
                                                  options))
      FATAL_ERROR("Result value mismatch!");
  }

  // Run actual timed test.
  options.verify    = false;  // Set verify to false for csv output.
  options.warm_up_n = warm_up_n_bak;
  options.n         = n_bak;
  fn(options, gemm_args);

  // Reset verify for next matrix size.
  options.verify = true;
}

/*************************** Internal setup fns **************************/
template <class scalar_type, class vta, class vtb, class vtc, class device_type>
gemm_args_t __do_setup(options_t options, matrix_dims_t dims) {
  using execution_space = typename device_type::execution_space;

  gemm_args_t gemm_args;
  uint64_t seed = Kokkos::Impl::clock_tic();
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
  STATUS;

  gemm_args.dims   = dims;
  gemm_args.transA = options.blas_args.gemm.gemm_args.c_str()[0];
  gemm_args.transB = options.blas_args.gemm.gemm_args.c_str()[1];
  if (options.test == BATCHED_TEAM_SIMD ||
      options.test == BATCHED_TEAM_SIMD_BLOCKED ||
      options.test == BATCHED_SERIAL_SIMD ||
      options.test == BATCHED_SERIAL_SIMD_BLOCKED ||
      options.test == BATCHED_SERIAL_COMPACT_MKL) {
    // Calculate the batch size for simd views
    auto a_simd_batch_size =
        dims.a.k / simd_vector_size + (dims.a.k % simd_vector_size > 0);
    auto b_simd_batch_size =
        dims.b.k / simd_vector_size + (dims.b.k % simd_vector_size > 0);
    auto c_simd_batch_size =
        dims.c.k / simd_vector_size + (dims.c.k % simd_vector_size > 0);

    // Reference gemm simd arguments for allocating A, B, and C matrices
    gemm_simd_args_t &A = gemm_args.Av, &B = gemm_args.Bv, &C = gemm_args.Cv;

    if (options.blas_args.batch_size_last_dim) {
      // Construct simd matrices with batch_size in the last dimension (better
      // for LayoutLeft views)
      A.vec_3d  = vector_view_type_3d("A_vector", dims.a.m, dims.a.n,
                                     a_simd_batch_size);
      A.mat_4d  = view_type_4d((scalar_type *)A.vec_3d.data(), simd_vector_size,
                              dims.a.m, dims.a.n, a_simd_batch_size);
      A.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)A.mat_4d.data(),
          simd_vector_size / simd_internal_vector_size, dims.a.m, dims.a.n,
          a_simd_batch_size);

      B.vec_3d  = vector_view_type_3d("B_vector", dims.b.m, dims.b.n,
                                     b_simd_batch_size);
      B.mat_4d  = view_type_4d((scalar_type *)B.vec_3d.data(), simd_vector_size,
                              dims.b.m, dims.b.n, b_simd_batch_size);
      B.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)B.mat_4d.data(),
          simd_vector_size / simd_internal_vector_size, dims.b.m, dims.b.n,
          b_simd_batch_size);

      C.vec_3d  = vector_view_type_3d("C_vector", dims.c.m, dims.c.n,
                                     c_simd_batch_size);
      C.mat_4d  = view_type_4d((scalar_type *)C.vec_3d.data(), simd_vector_size,
                              dims.c.m, dims.c.n, c_simd_batch_size);
      C.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)C.mat_4d.data(),
          simd_vector_size / simd_internal_vector_size, dims.c.m, dims.c.n,
          c_simd_batch_size);

    } else {
      // Construct simd matrices with batch_size in the first dimension (better
      // for LayoutRight views)
      A.vec_3d = vector_view_type_3d("A_vector", a_simd_batch_size, dims.a.m,
                                     dims.a.n);
      A.mat_4d = view_type_4d((scalar_type *)A.vec_3d.data(), a_simd_batch_size,
                              dims.a.m, dims.a.n, simd_vector_size);
      A.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)A.mat_4d.data(), a_simd_batch_size, dims.a.m,
          dims.a.n, simd_vector_size / simd_internal_vector_size);

      B.vec_3d = vector_view_type_3d("B_vector", b_simd_batch_size, dims.b.m,
                                     dims.b.n);
      B.mat_4d = view_type_4d((scalar_type *)B.vec_3d.data(), b_simd_batch_size,
                              dims.b.m, dims.b.n, simd_vector_size);
      B.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)B.mat_4d.data(), b_simd_batch_size, dims.b.m,
          dims.b.n, simd_vector_size / simd_internal_vector_size);

      C.vec_3d = vector_view_type_3d("C_vector", c_simd_batch_size, dims.c.m,
                                     dims.c.n);
      C.mat_4d = view_type_4d((scalar_type *)C.vec_3d.data(), c_simd_batch_size,
                              dims.c.m, dims.c.n, simd_vector_size);
      C.ivec_4d = internal_vector_view_type_4d(
          (internal_vector_type *)C.mat_4d.data(), c_simd_batch_size, dims.c.m,
          dims.c.n, simd_vector_size / simd_internal_vector_size);
    }

    // Use the non-simd 4-rank view type to randomly populate the gemm simd
    // arguments
    using tmp_view_type_4d =
        Kokkos::View<double ****, default_layout, default_device>;
    tmp_view_type_4d tmpA(
        "tmpA", gemm_args.Av.mat_4d.extent(0), gemm_args.Av.mat_4d.extent(1),
        gemm_args.Av.mat_4d.extent(2), gemm_args.Av.mat_4d.extent(3));
    Kokkos::fill_random(tmpA, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());
    tmp_view_type_4d tmpB(
        "tmpB", gemm_args.Bv.mat_4d.extent(0), gemm_args.Bv.mat_4d.extent(1),
        gemm_args.Bv.mat_4d.extent(2), gemm_args.Bv.mat_4d.extent(3));
    Kokkos::fill_random(tmpB, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());
    tmp_view_type_4d tmpC(
        "tmpC", gemm_args.Cv.mat_4d.extent(0), gemm_args.Cv.mat_4d.extent(1),
        gemm_args.Cv.mat_4d.extent(2), gemm_args.Cv.mat_4d.extent(3));
    Kokkos::fill_random(tmpC, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());
    Kokkos::fence();
    Kokkos::deep_copy(gemm_args.Av.mat_4d, tmpA);
    Kokkos::deep_copy(gemm_args.Bv.mat_4d, tmpB);
    Kokkos::deep_copy(gemm_args.Cv.mat_4d, tmpC);
    Kokkos::fence();
  } else {
    if (options.blas_args.batch_size_last_dim) {
      gemm_args.A = vta("gemm_args.A", dims.a.m, dims.a.n, dims.a.k);
      gemm_args.B = vtb("gemm_args.B", dims.b.m, dims.b.n, dims.b.k);
      gemm_args.C = vtc("gemm_args.C", dims.c.m, dims.c.n, dims.c.k);
    } else {
      gemm_args.A = vta("gemm_args.A", dims.a.k, dims.a.m, dims.a.n);
      gemm_args.B = vtb("gemm_args.B", dims.b.k, dims.b.m, dims.b.n);
      gemm_args.C = vtc("gemm_args.C", dims.c.k, dims.c.m, dims.c.n);
    }

    using tmp_view_type_3d =
        Kokkos::View<double ***, default_layout, default_device>;
    tmp_view_type_3d tmpA("tmpA", gemm_args.A.extent(0), gemm_args.A.extent(1),
                          gemm_args.A.extent(2));
    Kokkos::fill_random(tmpA, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());
    tmp_view_type_3d tmpB("tmpB", gemm_args.B.extent(0), gemm_args.B.extent(1),
                          gemm_args.B.extent(2));
    Kokkos::fill_random(tmpB, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());
    tmp_view_type_3d tmpC("tmpC", gemm_args.C.extent(0), gemm_args.C.extent(1),
                          gemm_args.C.extent(2));
    Kokkos::fill_random(tmpC, rand_pool,
                        Kokkos::rand<Kokkos::Random_XorShift64<execution_space>,
                                     double>::max());

    Kokkos::fence();
    Kokkos::deep_copy(gemm_args.A, tmpA);
    Kokkos::deep_copy(gemm_args.B, tmpB);
    Kokkos::deep_copy(gemm_args.C, tmpC);
    Kokkos::fence();
  }
  gemm_args.alpha         = options.blas_args.gemm.alpha;
  gemm_args.beta          = options.blas_args.gemm.beta;
  gemm_args.bp.team_size  = options.blas_args.team_size;
  gemm_args.bp.vector_len = options.blas_args.vector_len;
  gemm_args.bp.divisor    = options.blas_args.divisor;

  Kokkos::fence();  // Ensure that fill_random has completed.

  return gemm_args;
}

/*************************** Interal run helper fns **************************/
void __do_loop_and_invoke(options_t options,
                          void (*fn)(options_t, gemm_args_t)) {
  matrix_dims_t cur_dims;
  gemm_args_t gemm_args;
  STATUS;

  __print_gemm_perf_test_options(options);
  std::cout << "SCALAR:" << typeid(default_scalar).name()
            << ", LAYOUT:" << typeid(default_layout).name()
            << ", DEVICE:" << typeid(default_device).name()
            << ", SPACE:" << typeid(memory_space).name() << std::endl;

  options.out[0] << gemm_csv_header_str << std::endl;

  for (cur_dims = options.start;
       cur_dims.a.m <= options.stop.a.m && cur_dims.a.n <= options.stop.a.n &&
       cur_dims.b.m <= options.stop.b.m && cur_dims.b.n <= options.stop.b.n &&
       cur_dims.c.m <= options.stop.c.m && cur_dims.c.n <= options.stop.c.n;
       cur_dims.a.m += options.step, cur_dims.a.n += options.step,
      cur_dims.b.m += options.step, cur_dims.b.n += options.step,
      cur_dims.c.m += options.step, cur_dims.c.n += options.step) {
    gemm_args = __do_setup<default_scalar, view_type_3d, view_type_3d,
                           view_type_3d, default_device>(options, cur_dims);

    if (options.verify) {
      __gemm_do_verify<default_scalar, default_layout, default_device>(
          options, gemm_args, fn);
    } else {
      fn(options, gemm_args);
    }
  }
  return;
}

/*************************** External fns **************************/
void do_gemm_serial_blas(options_t options) {
  STATUS;
  __do_loop_and_invoke(
      options, __do_gemm_serial_blas<default_scalar, view_type_3d, view_type_3d,
                                     default_device>);
  return;
}

void do_gemm_serial_batched(options_t options) {
  STATUS;
  __do_loop_and_invoke(
      options, __do_gemm_serial_batched<default_scalar, view_type_3d,
                                        view_type_3d, view_type_3d,
                                        default_device, Algo::Gemm::Unblocked>);
  return;
}

void do_gemm_serial_batched_blocked(options_t options) {
  STATUS;
  __do_loop_and_invoke(
      options, __do_gemm_serial_batched<default_scalar, view_type_3d,
                                        view_type_3d, view_type_3d,
                                        default_device, Algo::Gemm::Blocked>);
  return;
}

void do_gemm_serial_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3Tag, Algo::Gemm::Unblocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<SerialTag, Algo::Gemm::Unblocked,
                                            default_device>);
  return;
}

void do_gemm_serial_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3Tag, Algo::Gemm::Blocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<SerialTag, Algo::Gemm::Blocked,
                                            default_device>);
  return;
}

void do_gemm_serial_opt1_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3TagOpt1,
                                   Algo::Gemm::Unblocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialTagOpt1, Algo::Gemm::Unblocked,
                                   default_device>);
  return;
}

void do_gemm_serial_opt1_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3TagOpt1, Algo::Gemm::Blocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<SerialTagOpt1, Algo::Gemm::Blocked,
                                            default_device>);
  return;
}

void do_gemm_serial_opt2_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim) {
    if (options.tile.m != 1 || options.tile.n != 1) {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialBatchDim3TagOpt2Tiled,
                                     Algo::Gemm::Unblocked, default_device>);
    } else {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialBatchDim3TagOpt2,
                                     Algo::Gemm::Unblocked, default_device>);
    }
  } else {
    if (options.tile.m != 1 || options.tile.n != 1) {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialTagOpt2Tiled, Algo::Gemm::Unblocked,
                                     default_device>);
    } else {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialTagOpt2, Algo::Gemm::Unblocked,
                                     default_device>);
    }
  }
  return;
}

void do_gemm_serial_opt2_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim) {
    if (options.tile.m != 1 || options.tile.n != 1) {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialBatchDim3TagOpt2Tiled,
                                     Algo::Gemm::Blocked, default_device>);
    } else {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialBatchDim3TagOpt2,
                                     Algo::Gemm::Blocked, default_device>);
    }
  } else {
    if (options.tile.m != 1 || options.tile.n != 1) {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialTagOpt2Tiled, Algo::Gemm::Blocked,
                                     default_device>);
    } else {
      __do_loop_and_invoke(
          options,
          __do_gemm_parallel_batched<SerialTagOpt2, Algo::Gemm::Blocked,
                                     default_device>);
    }
  }
  return;
}

void do_gemm_serial_optteam_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3TagOptTeam,
                                   Algo::Gemm::Unblocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialTagOptTeam, Algo::Gemm::Unblocked,
                                   default_device>);
  return;
}

void do_gemm_serial_optteam_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialBatchDim3TagOptTeam,
                                   Algo::Gemm::Blocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialTagOptTeam, Algo::Gemm::Blocked,
                                   default_device>);
  return;
}

void do_gemm_serial_simd_batched_parallel(options_t options) {
  STATUS;
  // SerialBatchDim3Tag
  // SerialSimdTag
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamSimdBatchDim4Tag, Algo::Gemm::Unblocked,
                                   default_device, Mode::Serial>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamSimdTag, Algo::Gemm::Unblocked,
                                            default_device, Mode::Serial>);
  return;
}

void do_gemm_serial_simd_batched_blocked_parallel(options_t options) {
  STATUS;
  // SerialBatchDim3Tag
  // SerialSimdTag
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamSimdBatchDim4Tag, Algo::Gemm::Blocked,
                                   default_device, Mode::Serial>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamSimdTag, Algo::Gemm::Blocked,
                                            default_device, Mode::Serial>);
  return;
}

void do_gemm_serial_batched_compact_mkl_parallel(options_t options) {
  STATUS;
#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialSimdBatchDim3Tag,
                                   Algo::Gemm::CompactMKL, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<SerialSimdTag, Algo::Gemm::CompactMKL,
                                   default_device>);
#else
#if !defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__)
  std::cerr
      << std::string(__func__)
      << " disabled since __KOKKOSBATCHED_ENABLE_INTEL_MKL__ is undefined."
      << std::endl;
#elif !defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__)
  std::cerr << std::string(__func__)
            << " disabled since __KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__ is "
               "undefined."
            << std::endl;
#elif !defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
  std::cerr
      << std::string(__func__)
      << " disabled since __KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__ "
         "is undefined."
      << std::endl;
#endif
#endif
  return;
}

void do_gemm_team_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3Tag, Algo::Gemm::Unblocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamTag, Algo::Gemm::Unblocked,
                                            default_device>);
  return;
}

void do_gemm_team_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3Tag, Algo::Gemm::Blocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamTag, Algo::Gemm::Blocked,
                                            default_device>);
  return;
}

void do_gemm_team_shmem_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamShmemBatchDim3Tag, Algo::Gemm::Unblocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamShmemTag, Algo::Gemm::Unblocked,
                                            default_device>);
  return;
}

void do_gemm_team_shmem_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamShmemBatchDim3Tag, Algo::Gemm::Blocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamShmemTag, Algo::Gemm::Blocked,
                                            default_device>);
  return;
}

void do_gemm_team_opt1_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3TagOpt1, Algo::Gemm::Unblocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamTagOpt1, Algo::Gemm::Unblocked,
                                            default_device>);
  return;
}

void do_gemm_team_opt1_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3TagOpt1, Algo::Gemm::Blocked,
                                   default_device>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamTagOpt1, Algo::Gemm::Blocked,
                                            default_device>);
  return;
}

void do_gemm_team_optdivisor_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3TagOptDivisor,
                                   Algo::Gemm::Unblocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamTagOptDivisor, Algo::Gemm::Unblocked,
                                   default_device>);
  return;
}

void do_gemm_team_optdivisor_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamBatchDim3TagOptDivisor,
                                   Algo::Gemm::Blocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamTagOptDivisor, Algo::Gemm::Blocked,
                                   default_device>);
  return;
}

void do_gemm_team_vector_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamVectorBatchDim3Tag,
                                   Algo::Gemm::Unblocked, default_device>);
  else
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamVectorTag, Algo::Gemm::Unblocked,
                                   default_device>);
  return;
}

void do_gemm_team_simd_batched_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamSimdBatchDim4Tag, Algo::Gemm::Unblocked,
                                   default_device, Mode::Team>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamSimdTag, Algo::Gemm::Unblocked,
                                            default_device, Mode::Team>);
  return;
}

void do_gemm_team_simd_batched_blocked_parallel(options_t options) {
  STATUS;
  if (options.blas_args.batch_size_last_dim)
    __do_loop_and_invoke(
        options,
        __do_gemm_parallel_batched<TeamSimdBatchDim4Tag, Algo::Gemm::Blocked,
                                   default_device, Mode::Team>);
  else
    __do_loop_and_invoke(
        options, __do_gemm_parallel_batched<TeamSimdTag, Algo::Gemm::Blocked,
                                            default_device, Mode::Team>);
  return;
}

// Blocked algo not yet implemented for TeamVectorGemm.
/* void do_gemm_team_vector_batched_blocked_parallel(options_t options) {
  STATUS;
  __do_loop_and_invoke(
      options, __do_gemm_parallel_batched<TeamVectorTag, Algo::Gemm::Blocked,
default_device>); return;
} */

void do_gemm_experiment_parallel(options_t options) {
  STATUS;
  using TransAType   = Trans::NoTranspose;
  using TransBType   = Trans::NoTranspose;
  using BlockingType = Algo::Gemm::Unblocked;

  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment1<TransAType, TransBType,
                                              BlockingType, default_device>);
  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment2<TransAType, TransBType,
                                              BlockingType, default_device>);
  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment3<TransAType, TransBType,
                                              BlockingType, default_device>);
  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment4<TransAType, TransBType,
                                              BlockingType, default_device>);
  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment5<TransAType, TransBType,
                                              BlockingType, default_device>);
  __do_loop_and_invoke(
      options, __do_gemm_parallel_experiment6<TransAType, TransBType,
                                              BlockingType, default_device>);
}

#endif  // KOKKOSBLAS3_GEMM_PERF_TEST_H_
