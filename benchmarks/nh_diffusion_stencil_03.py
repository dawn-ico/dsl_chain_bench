from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def mo_nh_diffusion_stencil_03(kh_smag_ec: Field[Edge, K], vn: Field[Edge, K],
                               e_bln_c_s: Field[Cell > Edge], geofac_div: Field[Cell > Edge], diff_multfac_smag: Field[K],
                               kh_c: Field[Cell, K], div: Field[Cell, K]):
    with domain.upward:
        kh_c = sum_over(Cell > Edge, kh_smag_ec*e_bln_c_s)/diff_multfac_smag
        div = sum_over(Cell > Edge, vn*geofac_div)
