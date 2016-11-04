macro(raja_add_benchmark name)
  add_test(
    NAME ${name}-test
    COMMAND ${name} --benchmark_format=json --benchmark_output=${name}-benchmark-results.json)
endmacro(raja_add_benchmark)
