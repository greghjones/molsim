#ifndef _MOLSIM_FUNCTIONAL_H
#define _MOLSIM_FUNCTIONAL_H

#include <utility>

#include "molsim_config.h"

#include "util.hpp"

enum vector_dispatch_t {
    none,
    neon,
    avx2,
    avx512
};

extern const vector_dispatch_t dispatchto;

namespace molsim
{

namespace functional
{
    namespace detail
    {
        #if defined(MOLSIM_ARM)
        void calc_tau_neon(const AlignedVector<double>& aij,
                           const AlignedVector<int>& gup,
                           const AlignedVector<double>& eup,
                           const AlignedVector<double>& frequencies,
                           const double columndensity,
                           const double Tex,
                           const double dV,
                           const double q,
                                 AlignedVector<double>& tau);
        void make_gaussians_neon(const AlignedVector<double>& centers,
                                 const AlignedVector<double>& int0s,
                                 const AlignedVector<ssize_t>& lls,
                                 const AlignedVector<ssize_t>& uls,
                                 const double dV,
                                 const AlignedVector<double>& freq_profile,
                                       AlignedVector<double>& tau_profile);
        void calc_Tb_neon(const AlignedVector<double>& frequency,
                          const AlignedVector<double>& tau,
                          const AlignedVector<double>& Tbg,
                          const double Tex,
                                AlignedVector<double>& Tb);
        void calc_Tb_neon(const AlignedVector<double>& frequency,
                          const AlignedVector<double>& tau,
                          const double Tbg,
                          const double Tex,
                                AlignedVector<double>& Tb);
        std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_neon(const AlignedVector<double>& freq_array,
                                                                               const AlignedVector<double>& int_arr,
                                                                               const double source_size,
                                                                               const double dish_size);
        #elif defined(MOLSIM_X86_64)
        void calc_tau_avx2(const AlignedVector<double>& aij,
                           const AlignedVector<int>& gup,
                           const AlignedVector<double>& eup,
                           const AlignedVector<double>& frequencies,
                           const double columndensity,
                           const double Tex,
                           const double dV,
                           const double q,
                                 AlignedVector<double>& tau);
        void make_gaussians_avx2(const AlignedVector<double>& centers,
                                 const AlignedVector<double>& int0s,
                                 const AlignedVector<ssize_t>& lls,
                                 const AlignedVector<ssize_t> uls,
                                 const double dV,
                                 const AlignedVector<double>& freq_profile,
                                       AlignedVector<double>& tau_profile);
        void calc_Tb_avx2(const AlignedVector<double>& frequency,
                          const AlignedVector<double>& tau,
                          const AlignedVector<double>& Tbg,
                          const double Tex,
                                AlignedVector<double>& Tb);
        void calc_Tb_avx2(const AlignedVector<double>& frequency,
                          const AlignedVector<double>& tau,
                          const double Tbg,
                          const double Tex,
                                AlignedVector<double>& Tb);
        std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_avx2(const AlignedVector<double>& freq_array,
                                                                               const AlignedVector<double>& int_arr,
                                                                               const double source_size,
                                                                               const double dish_size);
        #endif
        void calc_tau_purec(const AlignedVector<double>& aij,
                            const AlignedVector<int>& gup,
                            const AlignedVector<double>& eup,
                            const AlignedVector<double>& frequencies,
                            const double columndensity,
                            const double Tex,
                            const double dV,
                            const double q,
                                  AlignedVector<double>& tau);
        void make_gaussians_purec(const AlignedVector<double>& centers,
                                  const AlignedVector<double>& int0s,
                                  const AlignedVector<ssize_t>& lls,
                                  const AlignedVector<ssize_t> uls,
                                  const double dV,
                                  const AlignedVector<double>& freq_profile,
                                        AlignedVector<double>& tau_profile);
        void calc_Tb_purec(const AlignedVector<double>& frequency,
                           const AlignedVector<double>& tau,
                           const AlignedVector<double>& Tbg,
                           const double Tex,
                                 AlignedVector<double>& Tb);
        void calc_Tb_purec(const AlignedVector<double>& frequency,
                           const AlignedVector<double>& tau,
                           const double Tbg,
                           const double Tex,
                                 AlignedVector<double>& Tb);
        std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam_purec(const AlignedVector<double>& freq_array,
                                                                                const AlignedVector<double>& int_arr,
                                                                                const double source_size,
                                                                                const double dish_size);
    }

    void apply_vlsr(const AlignedVector<double>& infreq, const double vlsr, AlignedVector<double>& outfreq);
    static inline AlignedVector<double> apply_vlsr(const AlignedVector<double>& infreq, const double vlsr)
    {
        AlignedVector<double> out(infreq.size());
        apply_vlsr(infreq, vlsr, out);
        return out;
    };

    template <typename... Args>
    static inline void calc_tau(Args&&... args)
    {
        switch (dispatchto)
        {
            #if defined(MOLSIM_ARM)
            case neon:
                detail::calc_tau_neon(std::forward<Args>(args)...);
                break;
            #elif defined (MOLSIM_X86_64)
            case avx512:
            case avx2:
                detail::calc_tau_avx2(std::forward<Args>(args)...);
                break;
            #endif
            default:
                detail::calc_tau_purec(std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static inline void make_gaussians(Args&&... args)
    {
        switch (dispatchto)
        {
            #if defined(MOLSIM_ARM)
            case neon:
                detail::make_gaussians_neon(std::forward<Args>(args)...);
                break;
            #elif defined (MOLSIM_X86_64)
            case avx512:
            case avx2:
                detail::make_gaussians_avx2(std::forward<Args>(args)...);\
                break;
            #endif
            default:
                detail::make_gaussians_purec(std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static inline void calc_Tb(Args&&... args)
    {
        switch (dispatchto)
        {
            #if defined(MOLSIM_ARM)
            case neon:
                detail::calc_Tb_neon(std::forward<Args>(args)...);
                break;
            #elif defined (MOLSIM_X86_64)
            case avx512:
            case avx2:
                detail::calc_Tb_avx2(std::forward<Args>(args)...);
                break;
            #endif
            default:
                detail::calc_Tb_purec(std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static inline std::pair<AlignedVector<double>,AlignedVector<double>> apply_beam(Args&&... args)
    {
        switch (dispatchto)
        {
            #if defined(MOLSIM_ARM)
            case neon:
                return detail::apply_beam_neon(std::forward<Args>(args)...);
            #elif defined (MOLSIM_X86_64)
            case avx512:
            case avx2:
                return detail::apply_beam_avx2(std::forward<Args>(args)...);
            #endif
            default:
                return detail::apply_beam_purec(std::forward<Args>(args)...);
        }
    }
}
}

#ifndef BLAS
template<typename T>
static inline void __attribute__((always_inline)) axpby(long n, double alpha, const T* x, long incx, double beta, T* y, long incy)
{
    if (incx == 1 && incy == 1)
    {
        if (beta == 0)
        {
            #pragma omp simd
            for (long i = 0; i < n; i++)
                y[i] = alpha*x[i];
        }
        else
        {
            #pragma omp simd
            for (long i = 0; i < n; i++)
                y[i] = alpha*x[i] + beta*y[i];
        }
    }
    else
    {
        const long imax = n*incx;
        if (beta == 0)
        {
            long j = 0;
            #pragma omp simd
            for (long i = 0; i < imax; i += incx )
            {
                y[j] = alpha*x[i];
                j += incy;
            }
        }
        else
        {
            long j = 0;
            #pragma omp simd
            for (long i = 0; i < imax; i += incx)
            {
                y[j] = alpha*x[i] + beta*y[j];
                j += incy;
            }
        }
    }
}

template<typename T>
static inline void __attribute__((always_inline)) scal(long n, double alpha, T* x, long incx)
{
    if (incx == 1)
        #pragma omp simd
        for (long i = 0; i < n; i++)
            x[i] *= alpha;
    else
    {
        const long maxi = n*incx;
        for (long i = 0; i < maxi; i += incx)
            x[i] *= alpha;
    }
}
#endif

template <typename T, typename U, typename V>
static void __attribute__((always_inline)) vmulv(const AlignedVector<T>& a, const AlignedVector<U>& b, AlignedVector<V>& c, bool increment)
{
    assert(a.size() == b.size());
    assert(b.size() == c.size());

    if(increment)
    {
        #pragma omp simd
        for (long i = 0; i < a.size(); i++)
            c[i] += a[i]*b[i];
    }
    else
    {
        #pragma omp simd
        for (long i = 0; i < a.size(); i++)
            c[i] = a[i]*b[i];
    }
}

#endif
