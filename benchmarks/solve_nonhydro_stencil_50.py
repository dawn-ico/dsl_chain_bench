from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def mo_solve_nonhydro_stencil_50(z_q: Field[Cell, K], w: Field[Cell, K]):
    with domain.downward[1:-1] as k:
      w += w[k+1]*z_q