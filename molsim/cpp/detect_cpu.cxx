#include "molsim_config.h"
#include "functional.hpp"

static vector_dispatch_t detect_cpu()
{
    #if defined(MOLSIM_ARM)
        // armv8-a (minimum for aarch64) guarantees NEON and VFP
        return neon;
    #elif defined(MOLSIM_X86_64)
        #if defined(__GNUC__) || defined(__clang__)
            __builtin_cpu_init();
            if      (__builtin_cpu_supports("x86-64-v4"))
                return avx512;
            else if (__builtin_cpu_supports("x86-64-v3"))
                return avx2;
            else
                return none;
        #elif defined(_MSC_VER)
            if      (__check_isa_support(__IA_SUPPORT_VECTOR512))
                return avx512;
            else if (__check_isa_support(__IA_SUPPORT_VECTOR256))
                return avx2;
            else
                return none;
        #endif
    #else
    return none;
    #endif
}

const vector_dispatch_t dispatchto = detect_cpu();
