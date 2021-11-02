from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def unroll_e_v_e_inline(kh_smag_e: Field[Vertex, K], inv_dual_edge_length: Field[Vertex], theta_v: Field[Edge, K], outF: Field[Edge, K]):
    with domain.upward.across[nudging:halo]:
        outF = sum_over(Edge > Vertex, kh_smag_e*inv_dual_edge_length*sum_over(Vertex>Edge, theta_v))
