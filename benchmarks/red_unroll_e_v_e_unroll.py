from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def unroll_e_v_e_unroll(kh_smag_e: Field[Vertex, K], inv_dual_edge_length: Field[Vertex], theta_v: Field[Edge, K], z_temp: Field[Edge, K]):
    tmp: Field[Vertex, K]
    with domain.upward.across[nudging:halo]:
        tmp = sum_over(Vertex>Edge, theta_v)
        z_temp = sum_over(Edge > Vertex, kh_smag_e*inv_dual_edge_length*tmp)
