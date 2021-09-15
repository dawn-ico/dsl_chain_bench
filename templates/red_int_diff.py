from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def red_int_diff_{CHAIN_LETTERS}_{VERSION}(inF: Field[{CHAIN_2}, K], coeff: Field[{CHAIN_1} > {CHAIN_2}, K], 
                                          scale0: Field[{CHAIN_0}, K], scale1: Field[{CHAIN_0}, K], scale2: Field[{CHAIN_0}, K],
                                          outF: Field[{CHAIN_0}, K]):    
    tempF: Field[{CHAIN_1}, K]
    with domain.upward.across[nudging:halo]:
        tempF = sum_over({CHAIN_1} > {CHAIN_2}, inF*coeff)    
        outF = scale0*scale1*scale2*sum_over({CHAIN_0} > {CHAIN_1}, tempF)