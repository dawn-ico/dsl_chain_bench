from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def unroll_e_c_e_unroll(kh_smag_e: Field[Cell, K], inv_dual_edge_length: Field[Cell], theta_v: Field[Edge, K], outF: Field[Edge, K]):
    with domain.upward.across[nudging:halo]:
        outF = sum_over(Edge > Cell, kh_smag_e*inv_dual_edge_length*sum_over(Cell>Edge, theta_v))
