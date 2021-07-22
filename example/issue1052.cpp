#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>


int main(int argc, char *argv[]) {

  typedef double ST1;
  typedef float ST2;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Kokkos::View<ST1**,Kokkos::LayoutRight, EXSP> ViewDoubleType;
  typedef Kokkos::View<ST2**,Kokkos::LayoutRight, EXSP> ViewFloatType;

  Kokkos::initialize();
  {
    int n     = 4500;
    int iters = 1000;
    for (int i = 1; i < argc; ++i) {
      const std::string& token = argv[i];
      if (token == std::string("--size")) n = std::atoi(argv[++i]);
      if (token == std::string("--iters")) iters = std::atoi(argv[++i]);
      if (token == std::string("--help") || token == std::string("-h")) {
        std::cout
            << "Kokkos GMRES solver options:" << std::endl
            << "--size   :  The size 'n' of the long vector. (Default: 4500)."
            << std::endl
            << "--iters   :  The number of times to repeat call to gemm. (Default: 1000)."
            << std::endl
            << "--help  -h    :  Display this help message." << std::endl
            << "Example Call  :  ./ex_gemm.exe --size 300" << std::endl
            << std::endl;
        return 0;
      }
    }

    ViewDoubleType A(Kokkos::ViewAllocateWithoutInitializing("A"), n, 50);
    ViewDoubleType B(Kokkos::ViewAllocateWithoutInitializing("B"), 50, 50);
    ViewDoubleType C(Kokkos::ViewAllocateWithoutInitializing("C"), n, 50);

    ViewFloatType A2(Kokkos::ViewAllocateWithoutInitializing("A2"), n, 50);
    ViewFloatType B2(Kokkos::ViewAllocateWithoutInitializing("B2"), 50, 50);
    ViewFloatType C2(Kokkos::ViewAllocateWithoutInitializing("C2"), n, 50);

    int seed1 = 123;
    Kokkos::Random_XorShift64_Pool<> pool(seed1);
    Kokkos::fill_random(A, pool, -1, 1);
    Kokkos::fill_random(B, pool, -1, 1);
    Kokkos::fill_random(A2, pool, -1, 1);
    Kokkos::fill_random(B2, pool, -1, 1);
    int seed2 = 456;
    Kokkos::Random_XorShift64_Pool<> pool2(seed2);
    Kokkos::fill_random(C, pool2, -1, 1);
    Kokkos::fill_random(C2, pool2, -1, 1);

    cudaEvent_t start, stop, start2, stop2;

    // Try a warm-up loop:
    Kokkos::Tools::Experimental::pause_tools();
    for (int i = 0; i < 222; i++) {
      KokkosBlas::gemm("N", "N", 1.0, A, B, 1.0, C);
      KokkosBlas::gemm("N", "N", 1.0, A2, B2, 1.0, C2);
    }
    Kokkos::fence(); // ENsure that the warm-u[p kernels are done
    Kokkos::Tools::Experimental::resume_tools();

    float msecTotal = 0.0f;
    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    for (int i = 0; i < iters; i++) {
      KokkosBlas::gemm("N", "N", 1.0, A, B, 1.0, C);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    {
      checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
      // Compute and print the performance
      float msecPerMatrixMul   = msecTotal / iters;
      double flopsPerMatrixMul = 2.0 * (double)C.extent(0) * (double)C.extent(1) *
                                 (double)B.extent(0);  // TODO check this calc.
      double gigaFlops =
          (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
      printf("MatrixA(%lu,%lu), MatrixB(%lu,%lu), MatrixC(%lu,%lu)\n", A.extent(0), A.extent(1), B.extent(0), B.extent(1), C.extent(0), C.extent(1));
      printf("DOUBLE: Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
             gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);
    }

    Kokkos::fence(); // ENsure that the double gemm kernels are done

    msecTotal = 0.0f;
    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start2));
    checkCudaErrors(cudaEventCreate(&stop2));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start2, NULL));
    for(int i=0; i < iters; i++){
      KokkosBlas::gemm("N","N", 1.0, A2, B2, 1.0, C2);
    }
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop2, NULL));
    checkCudaErrors(cudaEventSynchronize(stop2));


    {
      checkCudaErrors(cudaEventElapsedTime(&msecTotal, start2, stop2));
      // Compute and print the performance
      float msecPerMatrixMul   = msecTotal / iters;
      double flopsPerMatrixMul = 2.0 * (double)C2.extent(0) * (double)C2.extent(1) *
                                 (double)B2.extent(0);  // TODO check this calc.
      double gigaFlops =
          (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
      printf("MatrixA2(%lu,%lu), MatrixB2(%lu,%lu), MatrixC2(%lu,%lu)\n", A2.extent(0), A2.extent(1), B2.extent(0), B2.extent(1), C2.extent(0), C2.extent(1));
      printf("FLOAT: Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
             gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);
    }

  }
  Kokkos::finalize();
}