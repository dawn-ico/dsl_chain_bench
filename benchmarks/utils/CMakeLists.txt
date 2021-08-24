add_library(thrustUtils STATIC
  thrustUtils.cu
)
set_source_files_properties(
      thrustUtils.cu
      PROPERTIES LANGUAGE CUDA)