from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def k_cache_sequential(z_w_con_c: Field[Cell, K], w: Field[Cell, K], coeff1_dwdz: Field[Cell, K], coeff2_dwdz: Field[Cell, K], ddt_w_adv: Field[Cell, K]):
    with domain.upward[1:].across[nudging:halo] as k:
        ddt_w_adv = -z_w_con_c*(w[k-1]*coeff1_dwdz - w[k+1]*coeff2_dwdz + w*(coeff2_dwdz - coeff1_dwdz))