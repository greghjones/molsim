subroutine make_gauss(centers, taus, l_idxs, u_idxs, freq_profile, tau_profile, npeaks, profilelength, dV, ckm)
    use, intrinsic :: iso_c_binding, only: c_double, c_long_long
    implicit none
    
    integer, parameter :: dp = c_double

    integer(c_long_long), intent(in) :: npeaks, profilelength
    real(dp), intent(in) :: dV, ckm
    integer(c_long_long), intent(in) :: l_idxs(npeaks), u_idxs(npeaks)
    real(dp), intent(in) :: centers(npeaks), taus(npeaks), freq_profile(profilelength)
    real(dp), intent(out) :: tau_profile(profilelength)

    integer(c_long_long) i, ll, ul
    real(dp) scale1, scale2, f0

    real(dp), parameter :: two = 2.0_dp
    real(dp), parameter :: sigma_to_fwhm = two*sqrt(two * log(two))

    scale1 = two*(dV/ckm/sigma_to_fwhm)**two

    do i=1,npeaks
        ll = l_idxs(i)
        ul = u_idxs(i)
        f0 = centers(i)
        scale2 = -1.0_dp / (scale1*f0*f0)
        tau_profile(ll:ul) = tau_profile(ll:ul) + taus(i)*exp(scale2*(freq_profile(ll:ul) - f0)**two)
    end do
    return
end subroutine make_gauss

subroutine calc_tb(freq_profile, tau_profile, tbg_profile, tex, h, k, tb_profile, profilelength, texlength)
    use, intrinsic :: iso_c_binding, only: c_double, c_long_long
    implicit none

    interface
        real(c_double) function expm1(x) bind(c, name='expm1')
            import, only : c_double
            real(c_double), intent(in), value :: x
        end function expm1
    end interface

    integer, parameter :: dp = c_double

    integer(c_long_long), intent(in) :: profilelength, texlength
    real(dp), intent(in) :: h, k
    real(dp), intent(in) :: freq_profile(profilelength), tau_profile(profilelength), tbg_profile(profilelength), tex(texlength)
    real(dp), intent(out) :: tb_profile(profilelength)

    integer(c_long_long) i

    real(dp) texv, scale, temp1, j_t, j_tbg

    scale = h*1.0d6/k

    if(texlength.eq.1)then
        texv = 1.0_dp/tex(1)
        do i=1,profilelength
            temp1 = scale*freq_profile(i)
            j_t   = temp1 / expm1(texv*temp1)
            j_tbg = temp1 / expm1(temp1/tbg_profile(i))
            tb_profile(i) = (j_tbg - j_t)*expm1(-tau_profile(i))
        enddo
    else
        do i=1,profilelength
            temp1 = scale*freq_profile(i)
            j_t   = temp1 / expm1(temp1/tex(i))
            j_tbg = temp1 / expm1(temp1/tbg_profile(i))
            tb_profile(i) = (j_tbg - j_t)*expm1(-tau_profile(i))
        enddo
    endif

end subroutine calc_tb

subroutine calc_tau(aij, gup, eup, frequencies, columndensity, tex, dV, q, h, k, cm, tau, ntransitions)
    use, intrinsic :: iso_c_binding, only: c_double, c_long_long
    implicit none

    interface
        real(c_double) function expm1(x) bind(c, name='expm1')
            import, only : c_double
            real(c_double), intent(in), value :: x
        end function expm1
    end interface

    integer, parameter :: dp = c_double

    integer(c_long_long), intent(in) :: ntransitions
    real(dp), intent(in) :: columndensity, tex, dV, q, h, k, cm
    real(dp), intent(in) :: aij(ntransitions), gup(ntransitions), eup(ntransitions), frequencies(ntransitions)
    real(dp), intent(out) :: tau(ntransitions)

    real(dp), parameter :: pi = 3.141592653589793238462643383279502884197_dp

    real(dp) :: prefactor, texinv, boltzmannscale
    real(dp) :: invfcubed(ntransitions), exp1(ntransitions), exp2(ntransitions)
    integer(c_long_long) i

    texinv = -1.0_dp/tex
    boltzmannscale = h*1.0d6/(k*tex)
    prefactor = log(2.0_dp)**0.5_dp * cm*cm*cm * (columndensity * 100.0_dp*100.0_dp) / (4*pi**1.5_dp * 1.0d18 * dV*1000.0_dp * q)

    exp1 = exp(texinv*eup)
    do i = 1,ntransitions
        exp2(i) = expm1(boltzmannscale*frequencies(i))
    enddo
    invfcubed = prefactor / (frequencies*frequencies*frequencies)

    tau = aij*gup*exp1*exp2*invfcubed

end subroutine calc_tau
