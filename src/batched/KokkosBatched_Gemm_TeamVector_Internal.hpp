#ifndef __KOKKOSBATCHED_GEMM_TEAMVECTOR_INTERNAL_HPP__
#define __KOKKOSBATCHED_GEMM_TEAMVECTOR_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Set_Internal.hpp"
#include "KokkosBatched_Scale_Internal.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector Internal Impl
  /// ==================== 
  template<typename ArgAlgo>
  struct TeamVectorGemmInternal {
    template<typename MemberType,
             typename ScalarType,
             typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, const int k,
           const ScalarType alpha, 
           const ValueType *__restrict__ A, const int as0, const int as1,
           const ValueType *__restrict__ B, const int bs0, const int bs1,
           const ScalarType beta,
           /**/  ValueType *__restrict__ C, const int cs0, const int cs1);
  };

  template<>
  template<typename MemberType,
           typename ScalarType,
           typename ValueType>
  KOKKOS_INLINE_FUNCTION
  int
  TeamVectorGemmInternal<Algo::Gemm::Unblocked>::
  invoke(const MemberType &member, 
         const int m, const int n, const int k,
         const ScalarType alpha, 
         const ValueType *__restrict__ A, const int as0, const int as1,
         const ValueType *__restrict__ B, const int bs0, const int bs1,
         const ScalarType beta,
         /**/  ValueType *__restrict__ C, const int cs0, const int cs1) {

    // C = beta C + alpha A B
    // C (m x n), A(m x k), B(k x n)
      
    const ScalarType one(1.0), zero(0.0);
        
    if      (beta == zero) TeamVectorSetInternal  ::invoke(member, m, n, zero, C, cs0, cs1);
    else if (beta != one ) TeamVectorScaleInternal::invoke(member, m, n, beta, C, cs0, cs1);

    if (alpha != ScalarType(0.0)) {
      if (m <= 0 || n <= 0 || k <= 0) return 0;

      if (beta != one) 
        member.team_barrier();

#if 1
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member,0,m*n),[&](const int &ij) {
        // assume layout right for batched computation
        const int i = ij/n, j = ij%n;
        const ValueType
          *__restrict__ pA = A+i*as0,
          *__restrict__ pB = B+j*bs1;
          
        ValueType c = ValueType(0);
        for (int p=0;p<k;++p) 
          c += pA[p*as1]*pB[p*bs0];
        C[i*cs0+j*cs1] += alpha*c;
      });
#endif
#if 0
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,m*n),[&](const int &ij) {
          // assume layout right for batched computation
          const int i = ij/n, j = ij%n;
          const ValueType
            *__restrict__ pA = A+i*as0,
            *__restrict__ pB = B+j*bs1;
            
          ValueType c = ValueType(0);
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member,k),[&](const int &p, ValueType &sum) {
            sum += pA[p*as1]*pB[p*bs0];
          },c);
          C[i*cs0+j*cs1] += alpha*c;
        });
#endif
#if 0
      ValueType
        *__restrict__ pC = C;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,k*m),[&](const int &km) {
        const int p = km / k, i = km % k;
        const ValueType
          *__restrict__ pA = A+p*as1,
          *__restrict__ pB = B+p*bs0;
        //for (int i=0;i<m;++i) {
        const ValueType tA(alpha*pA[i*as0]);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,n),[&](const int &j) {
          pC[i*cs0+j*cs1] += tA*pB[j*bs1];           
        });
      });
#endif
    }
    return 0;
  }

}


#endif
