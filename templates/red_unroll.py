from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def unroll_{CHAIN_LETTERS}_{VERSION}(kh_smag_e: Field[{CHAIN_1}, K], inv_dual_edge_length: Field[{CHAIN_1}], theta_v: Field[{CHAIN_2}, K], z_temp: Field[{CHAIN_0}, K]):   
    tmp: Field[{CHAIN_1}, K]
    with domain.upward.across[nudging:halo]:       
        tmp = sum_over({CHAIN_1}>{CHAIN_2}, theta_v)
        z_temp = sum_over({CHAIN_0} > {CHAIN_1}, kh_smag_e*inv_dual_edge_length*tmp)