from dusk.script import *


lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

dtime = Global("dtime")
wgt_nnow_vel = Global("wgt_nnow_vel")
wgt_nnew_vel = Global("wgt_nnew_vel")
cpd = Global("cpd")


@stencil
def mo_solve_nonhydro_stencil_40(
    z_w_expl: Field[Cell, K],
    w_nnow: Field[Cell, K],
    ddt_w_adv_ntl1: Field[Cell, K],
    ddt_w_adv_ntl2: Field[Cell, K],
    z_th_ddz_exner_c: Field[Cell, K],
    z_contr_w_fl_l: Field[Cell, K],
    rho_ic: Field[Cell, K],
    w_concorr_c: Field[Cell, K],
    vwind_expl_wgt: Field[Cell],
):

    with domain.upward[1:]:

        # explicit part for w - use temporally averaged advection terms for better numerical stability
        # the explicit weight for the pressure-gradient term is already included in z_th_ddz_exner_c
        z_w_expl = w_nnow + dtime * (
            wgt_nnow_vel * ddt_w_adv_ntl1
            + wgt_nnew_vel * ddt_w_adv_ntl2
            - cpd * z_th_ddz_exner_c
        )

        # contravariant vertical velocity times density for explicit part
        z_contr_w_fl_l = rho_ic * (-w_concorr_c + vwind_expl_wgt * w_nnow)