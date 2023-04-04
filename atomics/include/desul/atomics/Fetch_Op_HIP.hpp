/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FECH_OP_HIP_HPP_
#define DESUL_ATOMICS_FECH_OP_HIP_HPP_

#include <desul/atomics/Adapt_HIP.hpp>

namespace desul {
namespace Impl {

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, T)                           \
  template <class MemoryOrder, class MemoryScope>                       \
  __device__ inline T device_atomic_fetch_##OP(                         \
      T* ptr, T val, MemoryOrder, MemoryScope) {                        \
    return __hip_atomic_fetch_##OP(ptr,                                 \
                                   val,                                 \
                                   HIPMemoryOrder<MemoryOrder>::value,  \
                                   HIPMemoryScope<MemoryScope>::value); \
  }

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(OP) \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, int)           \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, long long)     \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, unsigned int)  \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, unsigned long long)

#define DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, float)               \
  DESUL_IMPL_HIP_ATOMIC_FETCH_OP(OP, double)

DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(add)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(min)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(max)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(and)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(or)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL(xor)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(add)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(min)
DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT(max)

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP_FLOATING_POINT
#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_HIP_ATOMIC_FETCH_OP

#define DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(T)                             \
  template <class MemoryOrder class MemoryScope>                       \
  __device__ inline T device_atomic_fetch_sub(                         \
      T* ptr, T val, MemoryOrder, MemoryScope) {                       \
    return __hip_atomic_fetch_add(ptr,                                 \
                                  -val,                                \
                                  HIPMemoryOrder<MemoryOrder>::value,  \
                                  HIPMemoryScope<MemoryScope>::value); \
  }

DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(int)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(long long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(unsigned int)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(unsigned long long)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(float)
DESUL_IMPL_HIP_ATOMIC_FETCH_SUB(double)

#undef DESUL_IMPL_HIP_ATOMIC_FETCH_SUB

// clang-format off
inline __device__                int device_atomic_fetch_inc(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1   ); }
inline __device__       unsigned int device_atomic_fetch_inc(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1u  ); }
inline __device__ unsigned long long device_atomic_fetch_inc(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1ull); }

inline __device__                int device_atomic_fetch_dec(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1  ); }
inline __device__       unsigned int device_atomic_fetch_dec(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1u ); }
inline __device__ unsigned long long device_atomic_fetch_dec(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -1  ); }

inline __device__       unsigned int device_atomic_fetch_inc_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicInc(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_dec_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicDec(ptr,  val); }
// clang-format on

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, TYPE)                                \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeDevice) {                           \
    __threadfence();                                                                   \
    TYPE return_val =                                                                  \
        device_atomic_fetch_##OP(ptr, val, MemoryOrderRelaxed(), MemoryScopeDevice()); \
    __threadfence();                                                                   \
    return return_val;                                                                 \
  }                                                                                    \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_fetch_##OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeCore) {                             \
    return device_atomic_fetch_##OP(ptr, val, MemoryOrder(), MemoryScopeDevice());     \
  }

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(OP) \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, int)           \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, unsigned int)  \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, unsigned long long)

#define DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(OP) \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, float)               \
  DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(OP, double)

DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(inc)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(dec)

DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(inc_mod, unsigned int)
DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP(dec_mod, unsigned int)

#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT
#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_HIP_DEVICE_ATOMIC_FETCH_OP

}  // namespace Impl
}  // namespace desul

#endif
