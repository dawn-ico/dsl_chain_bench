from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def unroll_c_e_c_unroll(kh_smag_e: Field[Edge, K], inv_dual_edge_length: Field[Edge], theta_v: Field[Cell, K], z_temp: Field[Cell, K]):
    with domain.upward.across[nudging:halo]:
        z_temp = sum_over(Cell > Edge, kh_smag_e*inv_dual_edge_length*sum_over(Edge>Cell, theta_v))
