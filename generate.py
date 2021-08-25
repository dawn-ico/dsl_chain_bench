from enum import Enum
import itertools

class Location(Enum):
  Edge = 0
  Cell = 1
  Vertex = 2

chains = [[Location.Edge, Location.Cell, Location.Vertex],
          [Location.Edge, Location.Cell, Location.Edge],
          [Location.Edge, Location.Vertex, Location.Cell],
          [Location.Edge, Location.Vertex, Location.Edge],
          [Location.Vertex, Location.Edge, Location.Cell],
          [Location.Vertex, Location.Edge, Location.Vertex],
          [Location.Vertex, Location.Cell, Location.Edge],
          [Location.Vertex, Location.Cell, Location.Vertex],
          [Location.Cell, Location.Edge, Location.Cell],
          [Location.Cell, Location.Edge, Location.Vertex],
          [Location.Cell, Location.Vertex, Location.Edge],
          [Location.Cell, Location.Vertex, Location.Cell]]

loc_to_char = {Location.Edge: "e", Location.Cell: "c", Location.Vertex: "v"}
loc_to_atlas = {Location.Edge: "edges()", Location.Cell: "cells()", Location.Vertex: "nodes()"}

def chain_to_letters(chain):
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc] + "_"
  return ret[:-1]

def chain_to_print(chain):  
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc].upper() + " > "
  return ret[:-3]

def fill_template(line, chain):
  line = line.replace("{CHAIN_LETTERS}", chain_to_letters(chain))
  line = line.replace("{CHAIN_PRINT}", chain_to_print(chain))
  line = line.replace("{CHAIN_0}", chain[0].name)
  line = line.replace("{CHAIN_1}", chain[1].name)
  line = line.replace("{CHAIN_2}", chain[2].name)
  line = line.replace("{CHAIN_0_MESH}", "mesh." + loc_to_atlas[chain[0]])
  line = line.replace("{CHAIN_1_MESH}", "mesh." + loc_to_atlas[chain[1]])
  line = line.replace("{CHAIN_2_MESH}", "mesh." + loc_to_atlas[chain[2]])
  return line

with open('templates/red_simple_bench.cpp', 'r') as bench_file, open('templates/red_simple.py', 'r') as sten_file:
  bench_lines = bench_file.readlines()
  bench_lines = [line.rstrip() for line in bench_lines]
  sten_lines = sten_file.readlines()
  sten_lines = [line.rstrip() for line in sten_lines]
  for chain in chains:
      with open('benchmarks/red_{}_bench.cpp'.format(chain_to_letters(chain)), "w+") as bench_out_file, open('benchmarks/red_{}.py'.format(chain_to_letters(chain)), "w+") as sten_out_file:
        for line in bench_lines:
          print(fill_template(line, chain), file=bench_out_file)
        for line in sten_lines:
          print(fill_template(line, chain), file=sten_out_file)
        