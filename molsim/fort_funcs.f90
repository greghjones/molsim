subroutine make_gauss(centers, taus, l_idxs, u_idxs, freq_profile, tau_profile, npeaks, profilelength, dV, ckm)
    implicit none
    integer*8, intent(in) :: npeaks, profilelength
    double precision, intent(in) :: dV, ckm
    integer*8, intent(in) :: l_idxs(npeaks), u_idxs(npeaks)
    double precision, intent(in) :: centers(npeaks), taus(npeaks), freq_profile(profilelength)
    double precision, intent(out) :: tau_profile(profilelength)

    integer*8 i, ll, ul
    double precision scale1, scale2, f0

    scale1 = 2.0d0*(dV/ckm/2.35482d0)**2.0d0

    do i=1,npeaks
        ll = l_idxs(i)
        ul = u_idxs(i)
        f0 = centers(i)
        scale2 = -1.0d0 / (scale1*f0*f0)
        tau_profile(ll:ul) = tau_profile(ll:ul) + taus(i)*exp(scale2*(freq_profile(ll:ul) - f0)**2.0d0)
    end do
    return
end subroutine make_gauss

subroutine calc_tb(freq_profile, tau_profile, tbg_profile, tex, h, k, tb_profile, profilelength, texlength)
    implicit none
    integer*8, intent(in) :: profilelength, texlength
    double precision, intent(in) :: h, k
    double precision, intent(in) :: freq_profile(profilelength), tau_profile(profilelength), tbg_profile(profilelength), tex(texlength)
    double precision, intent(out) :: tb_profile(profilelength)

    integer*8 i

    double precision texv, scale, temp1, j_t, j_tbg

    scale = h*1.0d6/k

    if(texlength.eq.1)then
        texv = 1.0d0/tex(1)
        do i=1,profilelength
            temp1 = scale*freq_profile(i)
            j_t   = 1.0d0 / (exp(texv*temp1)           - 1.0d0)
            j_tbg = 1.0d0 / (exp(temp1/tbg_profile(i)) - 1.0d0)
            tb_profile(i) = (j_t - j_tbg)*(1.0d0 - exp(-tau_profile(i)))
        enddo
    else
        do i=1,profilelength
            temp1 = scale*freq_profile(i)
            j_t   = 1.0d0 / (exp(temp1/tex(i))         - 1.0d0)
            j_tbg = 1.0d0 / (exp(temp1/tbg_profile(i)) - 1.0d0)
            tb_profile(i) = (j_t - j_tbg)*(1.0d0 - exp(-tau_profile(i)))
        enddo
    endif

end subroutine calc_tb

subroutine calc_tau(aij, gup, eup, frequencies, columndensity, tex, dV, q, h, k, cm, tau, ntransitions)
    implicit none
    integer*8, intent(in) :: ntransitions
    double precision, intent(in) :: columndensity, tex, dV, q, h, k, cm
    double precision, intent(in) :: aij(ntransitions), gup(ntransitions), eup(ntransitions), frequencies(ntransitions)
    double precision, intent(out) :: tau(ntransitions)

    double precision, parameter :: pi = 3.141592653589793238462643383279502884197d0

    double precision :: prefactor, texinv, boltzmannscale
    double precision :: invfcubed(ntransitions), exp1(ntransitions), exp2(ntransitions)

    texinv = -1.0d0/tex
    boltzmannscale = h*1.0d6/(k*tex)
    prefactor = log(2.0d0)**0.5d0 * cm*cm*cm * (columndensity * 100.0d0*100.0d0) / (4*pi**1.5d0 * 1.0d18 * dV*1000.0d0 * q)

    exp1 = exp(texinv*eup)
    exp2 = exp(boltzmannscale*frequencies) - 1.0d0
    invfcubed = prefactor / (frequencies*frequencies*frequencies)

    tau = aij*gup*exp1*exp2*invfcubed

end subroutine calc_tau

