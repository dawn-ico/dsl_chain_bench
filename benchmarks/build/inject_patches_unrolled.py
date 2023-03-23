from enum import Enum
import itertools
from shutil import copyfile

patch = """      
      int *{CHAIN_01_lower}Table_h = new int[{CHAIN_01_SIZE}*{CHAIN_0_Num}];
      int *{CHAIN_12_lower}Table_h = new int[{CHAIN_12_SIZE}*{CHAIN_1_Num}];
      int *{CHAIN_012_lower}Table_h = new int[{CHAIN_012_COMPRESSED_SIZE}*{CHAIN_0_Num}];
      int *{CHAIN_012_lower}TableTransposed_h = new int[{CHAIN_012_COMPRESSED_SIZE}*{CHAIN_0_Num}];

      cudaMemcpy({CHAIN_01_lower}Table_h, mesh_.{CHAIN_01_lower}Table, sizeof(int)*{CHAIN_01_SIZE}*{CHAIN_0_Num}, cudaMemcpyDeviceToHost);
      cudaMemcpy({CHAIN_12_lower}Table_h, mesh_.{CHAIN_12_lower}Table, sizeof(int)*{CHAIN_12_SIZE}*{CHAIN_1_Num}, cudaMemcpyDeviceToHost);                        

      std::fill({CHAIN_012_lower}Table_h, {CHAIN_012_lower}Table_h + {CHAIN_0_Num}*{CHAIN_012_COMPRESSED_SIZE}, -1);      

      std::unordered_map<int, int> nbh_counter;      
      for (int elemIdx = 0; elemIdx < {CHAIN_0_Num}; elemIdx++) {
        int lin_idx = 0;       
        nbh_counter.clear();
        for (int nbhIter0 = 0; nbhIter0 < {CHAIN_01_SIZE}; nbhIter0++) {
          int nbhIdx0 = {CHAIN_01_lower}Table_h[elemIdx + {CHAIN_0_Num} * nbhIter0];            
          if (nbhIdx0 == DEVICE_MISSING_VALUE) {
            continue;
          }          
          for (int nbhIter1 = 0; nbhIter1 < {CHAIN_12_SIZE}; nbhIter1++) {
            int nbhIdx1 = {CHAIN_12_lower}Table_h[nbhIdx0 + {CHAIN_1_Num} * nbhIter1];            
            if (nbhIdx1 >= 0 && nbh_counter.count(nbhIdx1) == 0) {
              nbh_counter[nbhIdx1] = 1;
            } else if (nbhIdx1 >= 0) {
              nbh_counter[nbhIdx1]++;
            }
            lin_idx++;
          }
        }       
        std::vector<std::pair<int, int>> elems(nbh_counter.begin(), nbh_counter.end());
        std::sort(elems.begin(), elems.end(), [](const std::pair<int, int>& left, const std::pair<int, int>& right) {return right.second < left.second;});	

        for (int linIdx = 0; linIdx < elems.size(); linIdx++) {
          {CHAIN_012_lower}Table_h[elemIdx + {CHAIN_0_Num}*linIdx] = elems[linIdx].first;          
        }        
      }

      cudaMalloc((void**) &mesh_.{CHAIN_012_lower}Table, sizeof(int)*{CHAIN_0_Num}*{CHAIN_012_COMPRESSED_SIZE});
      cudaMemcpy(mesh_.{CHAIN_012_lower}Table, {CHAIN_012_lower}Table_h, sizeof(int)*{CHAIN_0_Num}*{CHAIN_012_COMPRESSED_SIZE}, cudaMemcpyHostToDevice);

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());"""

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
loc_to_elem = {Location.Edge: "Edges", Location.Cell: "Cells", Location.Vertex: "Vertices"}

chain_to_weight_vec = {
  (Location.Edge, Location.Cell, Location.Edge):     "{2, 1, 1, 1, 1}",
  (Location.Edge, Location.Vertex, Location.Edge):   "{2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}",
  (Location.Edge, Location.Cell, Location.Vertex):   "{2, 2, 1, 1}",
  (Location.Edge, Location.Vertex, Location.Cell):   "{2, 2, 1, 1, 1, 1, 1, 1, 1, 1}",
  (Location.Cell, Location.Edge, Location.Cell):     "{3, 1, 1, 1}",
  (Location.Cell, Location.Vertex, Location.Cell):   "{3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1}",
  (Location.Cell, Location.Edge, Location.Vertex):   "{2, 2, 2}",
  (Location.Cell, Location.Vertex, Location.Edge):   "{2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}",
  (Location.Vertex, Location.Edge, Location.Vertex): "{6, 1, 1, 1, 1, 1, 1}",
  (Location.Vertex, Location.Cell, Location.Vertex): "{6, 2, 2, 2, 2, 2, 2}",
  (Location.Vertex, Location.Edge, Location.Cell):   "{2, 2, 2, 2, 2, 2}",
  (Location.Vertex, Location.Cell, Location.Edge):   "{2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1}",
}

chain_to_weight_vec_length =  {
  (Location.Edge, Location.Cell, Location.Edge):     "5",
  (Location.Edge, Location.Vertex, Location.Edge):   "12",
  (Location.Edge, Location.Cell, Location.Vertex):   "4",
  (Location.Edge, Location.Vertex, Location.Cell):   "10",
  (Location.Cell, Location.Edge, Location.Cell):     "4",
  (Location.Cell, Location.Vertex, Location.Cell):   "13",
  (Location.Cell, Location.Edge, Location.Vertex):   "3",
  (Location.Cell, Location.Vertex, Location.Edge):   "15",
  (Location.Vertex, Location.Edge, Location.Vertex): "7",
  (Location.Vertex, Location.Cell, Location.Vertex): "7",
  (Location.Vertex, Location.Edge, Location.Cell):   "6",
  (Location.Vertex, Location.Cell, Location.Edge):   "12",  
}

def chain_to_letters(chain):
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc] + "_"
  return ret[:-1]

def chain_to_lower(chain):
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc]
  return ret

def chain_to_print(chain):  
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc].upper() + " > "
  return ret[:-3] 

def chain_to_size(chain):
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc].upper() + "_"  
  return ret + "SIZE"

def chain_to_compressed_size(chain):
  ret = ""
  for loc in chain:
    ret += loc_to_char[loc].upper() + "_"  
  return ret + "COMPRESSED_SIZE"  

def elem_to_num(elem):
  return "Num" + loc_to_elem[elem]

def sparse_size(chain_left, chain_right):
  sizes = {(Location.Edge, Location.Cell)   : "2",
           (Location.Edge, Location.Vertex) : "2",
           (Location.Cell, Location.Vertex) : "3",
           (Location.Cell, Location.Edge)   : "3",
           (Location.Vertex, Location.Edge) : "6",
           (Location.Vertex, Location.Cell) : "6"}
  return sizes[(chain_left, chain_right)]


def fill_template(template, chain):
  template = template.replace("{CHAIN_01_lower}", chain_to_lower(chain[0:2]))
  template = template.replace("{CHAIN_12_lower}", chain_to_lower(chain[1:3]))
  template = template.replace("{CHAIN_012_lower}", chain_to_lower(chain))
  template = template.replace("{CHAIN_0_Num}", "mesh_." + elem_to_num(chain[0]))
  template = template.replace("{CHAIN_1_Num}", "mesh_." + elem_to_num(chain[1]))
  template = template.replace("{CHAIN_01_SIZE}", chain_to_size(chain[0:2]))
  template = template.replace("{CHAIN_12_SIZE}", chain_to_size(chain[1:3]))
  template = template.replace("{CHAIN_012_COMPRESSED_SIZE}", chain_to_compressed_size(chain))
  return template

for chain in chains:
  my_patch = fill_template(patch, chain)
  fname_in = "red_chain_{}_inline.cpp".format(chain_to_letters(chain))
  fname_out = "red_chain_{}_inline_patched.cpp".format(chain_to_letters(chain))
  print(fname_in)
  with open(fname_in, "r") as fin:
    lines = fin.readlines()              

    # put in the patch     
    for i, line in enumerate(lines):
      if line.strip().startswith("stream_ = stream;"):        
        lines.insert(i+1, my_patch)
        break

    # add additional includes
    for i, line in enumerate(lines):
      if line.strip().startswith("#include <chrono>"):        
        lines.insert(i, "#include <unordered_map>\n")
        lines.insert(i, "#include <algorithm>\n")
        break

    # add additional sizes    
    for i, line in enumerate(lines):
      if line.strip().startswith("static const int"):        
        lines.insert(i, "static const int " + chain_to_compressed_size(chain) + " = " + chain_to_weight_vec_length[tuple(chain)] + ";\n")
        lines.insert(i, "static const int " + chain_to_size(chain[0:2]) + " = " + sparse_size(chain[0], chain[1]) + ";\n")
        lines.insert(i, "static const int " + chain_to_size(chain[1:3]) + " = " + sparse_size(chain[1], chain[2]) + ";\n")
        break

    # add additional tables
    for i, line in enumerate(lines):
      if line.strip().startswith("dawn::unstructured_domain DomainUpper;"):        
        lines.insert(i+1, "    int *" + chain_to_lower(chain[0:2])+ "Table;\n")
        lines.insert(i+1, "    int *" + chain_to_lower(chain[1:3])+ "Table;\n")        
        break    
    for i, line in enumerate(lines):
      if line.strip().startswith(chain_to_lower(chain) + "Table = "):
        fstring = """    {}Table = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{{{{{}, {}}}, 0}})"""
        nested_table_idx = i
        lines.insert(i, fstring.format(chain_to_lower(chain[0:2]), "dawn::LocationType::" + loc_to_elem[chain[0]], "dawn::LocationType::" + loc_to_elem[chain[1]]) + ";\n")
        lines.insert(i, fstring.format(chain_to_lower(chain[1:3]), "dawn::LocationType::" + loc_to_elem[chain[1]], "dawn::LocationType::" + loc_to_elem[chain[2]]) + ";\n")
        break

    # remove old chained table. in brutal ways.
    chain_line_idx = -1
    for i, line in enumerate(lines):
      if line.strip().startswith(chain_to_lower(chain) + "Table = "):
        chain_line_idx = i
    lines[chain_line_idx] = "\n"
    lines[chain_line_idx+1] = "\n"
    if lines[chain_line_idx+2].strip() is not "}":
      lines[chain_line_idx+2] = "\n"

    # inject weights
    weight_line = -1
    for i, line in enumerate(lines):
      if line.strip().startswith("::dawn::float_type weights_"):
        weight_id = line[line.index("weights_"):line.index("weights_")+len("weights_")+2]
        weight_line = i
        break
    lines[i] = "    ::dawn::float_type " + weight_id +"["+ chain_to_weight_vec_length[tuple(chain)] +"] = "+ chain_to_weight_vec[tuple(chain)] + ";\n"

    # set template to compressed size
    templ_size = "<" + chain_to_size(chain) + ">"    
    for i, line in enumerate(lines):
      if line.find(templ_size) is not -1:
        launch_line = line.replace(templ_size, "<" + chain_to_compressed_size(chain)  + ">")
        launch_line_idx = i
        break
    lines[launch_line_idx] = launch_line

  with open(fname_out, "w+") as fout:
    for line in lines:
      print(line, file=fout, end='')
  copyfile(fname_out, fname_in)  