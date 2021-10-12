from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

dtime = Global("dtime")
wgt_nnow_rth = Global("wgt_nnow_rth")
wgt_nnew_rth = Global("wgt_nnew_rth")

@stencil
def mo_solve_nonhydro_stencil_10(w: Field[Cell, K], w_concorr_c: Field[Cell, K], ddqz_z_half: Field[Cell, K],
                                 rho_now: Field[Cell, K], rho_var: Field[Cell, K],
                                 theta_now: Field[Cell, K], theta_var: Field[Cell, K],
                                 wgtfac_c: Field[Cell, K], theta_ref_mc: Field[Cell, K],
                                 vwind_expl_wgt: Field[Cell], exner_pr: Field[Cell, K], d_exner_dz_ref_ic: Field[Cell, K],
                                 rho_ic: Field[Cell, K], z_theta_v_pr_ic: Field[Cell, K], theta_v_ic: Field[Cell, K], z_th_ddz_exner_c: Field[Cell, K]):
    z_w_backtraj: Field[Cell, K]
    z_rho_tavg_m1: Field[Cell, K]
    z_theta_tavg_m1: Field[Cell, K]
    z_rho_tavg: Field[Cell, K]
    z_theta_tavg: Field[Cell, K]
    z_theta_v_pr_mc_m1: Field[Cell, K]
    z_theta_v_pr_mc: Field[Cell, K]
    with domain.upward[1:] as k:
      # z_w_backtraj = - (p_nh%prog(nnew)%w(jc,jk,jb) - p_nh%diag%w_concorr_c(jc,jk,jb)) * &
      #   dtime*0.5_wp/p_nh%metrics%ddqz_z_half(jc,jk,jb)
      z_w_backtraj = - (w - w_concorr_c) * dtime*0.5/ddqz_z_half

      # z_rho_tavg_m1 = wgt_nnow_rth*p_nh%prog(nnow)%rho(jc,jk-1,jb) + &
      #                 wgt_nnew_rth*p_nh%prog(nvar)%rho(jc,jk-1,jb)
      # z_theta_tavg_m1 = wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,jk-1,jb) + &
      #                   wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,jk-1,jb)
      z_rho_tavg_m1 = wgt_nnow_rth*rho_now[k-1] + wgt_nnew_rth*rho_var[k-1]
      z_theta_tavg_m1 = wgt_nnow_rth*theta_now[k-1] + wgt_nnew_rth*theta_var[k-1]

      # z_rho_tavg = wgt_nnow_rth*p_nh%prog(nnow)%rho(jc,jk,jb) + &
      #              wgt_nnew_rth*p_nh%prog(nvar)%rho(jc,jk,jb)
      # z_theta_tavg = wgt_nnow_rth*p_nh%prog(nnow)%theta_v(jc,jk,jb) + &
      #                wgt_nnew_rth*p_nh%prog(nvar)%theta_v(jc,jk,jb)
      z_rho_tavg = wgt_nnow_rth*rho_now + wgt_nnew_rth*rho_var
      z_theta_tavg = wgt_nnow_rth*theta_now + wgt_nnew_rth*theta_var

      # p_nh%diag%rho_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *z_rho_tavg    + &
      #                       (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_rho_tavg_m1 + &
      #   z_w_backtraj*(z_rho_tavg_m1-z_rho_tavg)
      rho_ic = wgtfac_c*z_rho_tavg + (1 - wgtfac_c)*z_rho_tavg_m1 + z_w_backtraj*(z_rho_tavg_m1-z_rho_tavg)

      # z_theta_v_pr_mc_m1  = z_theta_tavg_m1 - p_nh%metrics%theta_ref_mc(jc,jk-1,jb)
      # z_theta_v_pr_mc     = z_theta_tavg    - p_nh%metrics%theta_ref_mc(jc,jk,jb)
      z_theta_v_pr_mc_m1  = z_theta_tavg_m1 - theta_ref_mc[k-1]
      z_theta_v_pr_mc     = z_theta_tavg    - theta_ref_mc[k]

      # z_theta_v_pr_ic(jc,jk) =                                       &
      #          p_nh%metrics%wgtfac_c(jc,jk,jb) *z_theta_v_pr_mc +    &
      #   (1._vp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_theta_v_pr_mc_m1
      z_theta_v_pr_ic = wgtfac_c *z_theta_v_pr_mc + (1-wgtfac_c)*z_theta_v_pr_mc_m1

      # p_nh%diag%theta_v_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *z_theta_tavg    +  &
      #                           (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*z_theta_tavg_m1 +  &
      #   z_w_backtraj*(z_theta_tavg_m1-z_theta_tavg)
      theta_v_ic = wgtfac_c *z_theta_tavg + (1-wgtfac_c)*z_theta_tavg_m1 +  z_w_backtraj*(z_theta_tavg_m1-z_theta_tavg)

      # z_th_ddz_exner_c(jc,jk,jb) = p_nh%metrics%vwind_expl_wgt(jc,jb)* &
      #   p_nh%diag%theta_v_ic(jc,jk,jb) * (p_nh%diag%exner_pr(jc,jk-1,jb)-      &
      #   p_nh%diag%exner_pr(jc,jk,jb)) / p_nh%metrics%ddqz_z_half(jc,jk,jb) +   &
      #   z_theta_v_pr_ic(jc,jk)*p_nh%metrics%d_exner_dz_ref_ic(jc,jk,jb)
      z_th_ddz_exner_c = vwind_expl_wgt * theta_v_ic * (exner_pr[k-1] - exner_pr[k]) / ddqz_z_half + z_theta_v_pr_ic*d_exner_dz_ref_ic