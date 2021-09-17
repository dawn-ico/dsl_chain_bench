from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def red_chain_{CHAIN_LETTERS}_{VERSION}(inF: Field[{CHAIN_2},K], outF: Field[{CHAIN_0},K]):
  with domain.across[nudging:halo].upward:      
    outF = sum_over({CHAIN_0} > {CHAIN_1} > {CHAIN_2}, inF[{CHAIN_0} > {CHAIN_1} > {CHAIN_2}], weights=[-1.])   