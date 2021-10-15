from enum import Enum
import itertools

NAME="unroll"

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

versions = ["inline", "sequential"]

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

def sparse_size(chain_left, chain_right):
  sizes = {(Location.Edge, Location.Cell)   : 2,
           (Location.Edge, Location.Vertex) : 2,
           (Location.Cell, Location.Vertex) : 3,
           (Location.Cell, Location.Edge)   : 3,
           (Location.Vertex, Location.Edge) : 6,
           (Location.Vertex, Location.Cell) : 6}
  return sizes[(chain_left, chain_right)]


def fill_template(line, chain, version=""):
  line = line.replace("{CHAIN_LETTERS}", chain_to_letters(chain))
  line = line.replace("{CHAIN_PRINT}", chain_to_print(chain))
  line = line.replace("{CHAIN_0}", chain[0].name)
  line = line.replace("{CHAIN_1}", chain[1].name)
  line = line.replace("{CHAIN_2}", chain[2].name)
  line = line.replace("{CHAIN_0_MESH}", "mesh." + loc_to_atlas[chain[0]])
  line = line.replace("{CHAIN_1_MESH}", "mesh." + loc_to_atlas[chain[1]])
  line = line.replace("{CHAIN_2_MESH}", "mesh." + loc_to_atlas[chain[2]])
  line = line.replace("{VERSION}", version)  
  line = line.replace("{SPARSE_12}", str(sparse_size(chain[1],chain[2])))
  return line

with open('templates/red_{}_bench.cpp'.format(NAME), 'r') as bench_file, open('templates/red_{}.py'.format(NAME), 'r') as sten_file:
  bench_lines = bench_file.readlines()
  bench_lines = [line.rstrip() for line in bench_lines]
  sten_lines = sten_file.readlines()
  sten_lines = [line.rstrip() for line in sten_lines]
  for chain in chains:
      if chain[0] is not chain[2]:
        continue
      with open('benchmarks/red_{}_{}_bench.cpp'.format(NAME, chain_to_letters(chain)), "w+") as bench_out_file:
        for line in bench_lines:
          print(fill_template(line, chain), file=bench_out_file)
        with open('benchmarks/red_{}_{}_inline.py'.format(NAME, chain_to_letters(chain)), "w+") as sten_out_inl_file, \
              open('benchmarks/red_{}_{}_sequential.py'.format(NAME, chain_to_letters(chain)), "w+") as sten_out_seq_file:
          for line in sten_lines:
            print(fill_template(line, chain, "inline"), file=sten_out_inl_file)
          for line in sten_lines:
            print(fill_template(line, chain, "sequential"), file=sten_out_seq_file)
        