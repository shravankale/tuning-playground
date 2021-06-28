/**
 * simple_features
 *
 * Complexity: simple
 *
 * Tuning problem:
 *
 * This example exists for two reasons. First: to show what a simple 
 * direct usage of the Tuning API looks like, and second, to provide
 * tools with an example of a simple problem with features.
 *
 * In this contrived example, you're asked to guess the cuteness of an
 * entity, along with what species it is. From 0, cuteness goes to 11.
 * Values 0-9 are reserved for species "person," while 10 and 11 are for
 * "dog." You're asked to return basically the same answer as you were given,
 * and penalized for a miss on cuteness or species.
 *
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
auto make_cuteness_candidates() {
  std::vector<int64_t> candidates{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int64_t *bad_candidate_impl =
      (int64_t *)malloc(sizeof(int64_t) * candidates.size());
  memcpy(bad_candidate_impl, candidates.data(),
         sizeof(int64_t) * candidates.size());
  return Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),
                                                         bad_candidate_impl);
}
int main(int argc, char *argv[]) {
  constexpr const int data_size = 1000;
  std::vector<std::string> species = {"dog", "person"};
  tuned_kernel(
      argc, argv,
      [&](const int total_iters) {
        size_t cuteness_percent_id;
        size_t is_dog_id;
        size_t c_answer_id;
        size_t s_answer_id;
        Kokkos::Tools::Experimental::VariableInfo cuteness_info;
        cuteness_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        cuteness_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_ratio;
        cuteness_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_range;
        cuteness_info.candidates = make_cuteness_candidates();
        Kokkos::Tools::Experimental::VariableInfo is_dog_info;
        is_dog_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_string;
        is_dog_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_categorical;
        is_dog_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        is_dog_info.candidates =
            Kokkos::Tools::Experimental::make_candidate_set(species.size(),
                                                            species.data());
        Kokkos::Tools::Experimental::VariableInfo c_answer_info;
        c_answer_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
        c_answer_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_ratio;
        c_answer_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        c_answer_info.candidates = make_cuteness_candidates();

        Kokkos::Tools::Experimental::VariableInfo s_answer_info;
        s_answer_info.type =
            Kokkos::Tools::Experimental::ValueType::kokkos_value_string;
        s_answer_info.category = Kokkos::Tools::Experimental::
            StatisticalCategory::kokkos_value_categorical;
        s_answer_info.valueQuantity =
            Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
        s_answer_info.candidates =
            Kokkos::Tools::Experimental::make_candidate_set(species.size(),
                                                            species.data());
        cuteness_percent_id = Kokkos::Tools::Experimental::declare_input_type(
            "tuning_playground.cuteness", cuteness_info);
        is_dog_id = Kokkos::Tools::Experimental::declare_input_type(
            "tuning_playground.is_dog", is_dog_info);
        c_answer_id = Kokkos::Tools::Experimental::declare_output_type(
            "tuning_playground.c_answer", c_answer_info);
        s_answer_id = Kokkos::Tools::Experimental::declare_output_type(
            "tuning_playground.s_answer", s_answer_info);

        return std::make_tuple(cuteness_percent_id, is_dog_id, c_answer_id,
                               s_answer_id);
      },
      [&](const int x, const int total_iters, size_t cuteness_id,
          size_t is_dog_id, size_t c_answer_id, size_t s_answer_id) {
        int64_t cuteness = x % 12;
        std::string name = (cuteness >= 10) ? "dog" : "person";
        std::vector<Kokkos::Tools::Experimental::VariableValue> feature_vector{
            Kokkos::Tools::Experimental::make_variable_value(cuteness_id,
                                                             cuteness),
            Kokkos::Tools::Experimental::make_variable_value(is_dog_id, name)};
        std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
            Kokkos::Tools::Experimental::make_variable_value(c_answer_id,
                                                             int64_t(0)),
            Kokkos::Tools::Experimental::make_variable_value(s_answer_id,
                                                             "person")};
        size_t context = Kokkos::Tools::Experimental::get_new_context_id();
        Kokkos::Tools::Experimental::begin_context(context);
        Kokkos::Tools::Experimental::set_input_values(context, 2,
                                                      feature_vector.data());
        Kokkos::Tools::Experimental::request_output_values(
            context, 2, answer_vector.data());
        auto penalty =
            (1000 * std::abs(cuteness - answer_vector[0].value.int_value)) +
            100 * ((strcmp(name.c_str(),
                           answer_vector[1].value.string_value)) == 0
                       ? 0
                       : 1);
        usleep(1 * penalty);
        Kokkos::Tools::Experimental::end_context(context);
      });
}
