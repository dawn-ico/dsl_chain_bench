from dusk.script import *

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def red_e_c_v_sequential(inF: Field[Vertex], outF: Field[Edge]):
  tempF: Field[Cell]
  with domain.across[nudging:halo].upward:
    tempF = sum_over(Cell > Vertex, inF)
    outF = sum_over(Edge > Cell, tempF)
