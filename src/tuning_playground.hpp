#ifndef TUNINGPLAYRGROUND_HPP
#define TUNINGPLAYRGROUND_HPP

#include<Kokkos_Core.hpp>
#include<unordered_map>
template<typename Setup, typename Tunable>
void tuned_kernel(int argc, char* argv[], Setup setup, Tunable tunable){
  int num_iters = 100000;
  bool tuned_internals;
  bool found_tuning_tool;
  bool print_progress;
  Kokkos::initialize(argc, argv);
  {
    auto kernel_data = setup(num_iters);
    for(int x =0; x < num_iters; ++x) {
      tunable(x, num_iters, kernel_data);
    } 
  
  }
  Kokkos::finalize();
}

void fastest_of_helper(int index){
  /** error case*/
}

template<typename Head, typename... Cons>
void fastest_of_helper(int index, Head head, Cons... cons){
  if(index == 0){
    return head();
  }
  return fastest_of_helper(index-1, cons...);
}

static std::unordered_map<std::string, size_t> ids_for_kernels;
size_t create_categorical_int_tuner(std::string name, size_t num_options){
  using namespace Kokkos::Tools::Experimental;
  VariableInfo info;
  info.category = StatisticalCategory::kokkos_value_categorical;
  info.type = ValueType::kokkos_value_int64;
  info.valueQuantity = CandidateValueType::kokkos_value_set;
  std::vector<int64_t> options;
  for(int x=0;x<num_options;++x){
    options.push_back(x);
  }
  info.candidates = make_candidate_set(options.size(), options.data());
  return declare_output_type(name, info);
}

size_t create_fastest_implementation_id(){
  using namespace Kokkos::Tools::Experimental;
  static size_t id;
  static bool done;
  if(!done){
    done = true;
    VariableInfo info;
    info.category = StatisticalCategory::kokkos_value_categorical;
    info.type = ValueType::kokkos_value_string;
    info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
    id = declare_input_type("playground.fastest_implementation_of", info);
  }
  return id;
}

template<typename ... Implementations>
void fastest_of(const std::string& label, Implementations... implementations){
    using namespace Kokkos::Tools::Experimental;
    auto tuner_iter = [&]() {
      auto my_tuner = ids_for_kernels.find(label);
      if (my_tuner == ids_for_kernels.end()) {
        return (ids_for_kernels.emplace(label, create_categorical_int_tuner(label, sizeof...(Implementations)))
                    .first);
      }
      return my_tuner;
    }();
    auto var_id = tuner_iter->second;
    auto input_id = create_fastest_implementation_id();
    VariableValue picked_implementation = make_variable_value(var_id,int64_t(0));
    VariableValue which_kernel = make_variable_value(var_id,label.c_str());
    auto context_id = get_new_context_id();
    begin_context(context_id);
    set_input_values(context_id, 1, &which_kernel);
    request_output_values(context_id, 1, &picked_implementation);
    fastest_of_helper(picked_implementation.value.int_value, implementations...);
    end_context(context_id);
}


#endif
