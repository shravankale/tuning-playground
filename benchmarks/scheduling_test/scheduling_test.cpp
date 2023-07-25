
/*
    'tuning_playground' plugin for testing Kokkos Scheduling for two loops that sleep only for a triangular matrix

    Values for the above varaibles:
    1. schedule (fixed) = Kokkos provides support only for dynamic and static scheduling

    Dimensions of matrices: M,N,P
*/

#include <tuning_playground.hpp>
#include <omp.h>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
#include <ctime>
#include <random>
#include <unistd.h>

const int M=16;
const int N=16;

template <typename T>
void doParallel(){

        
        /*Kokkos::parallel_for(re.size(), KOKKOS_LAMBDA(const int i) {
            double sleepDuration = static_cast<double>(i) / 1e6;
            //switch to usleep
            sleep(sleepDuration);
        });*/

        Kokkos::RangePolicy<Kokkos::OpenMP, T> outer_policy(0,M);

        Kokkos::parallel_for("outer_loop",outer_policy, KOKKOS_LAMBDA(const int i){

            //Kokkos::RangePolicy<Kokkos::OpenMP> inner_policy(i,N);

            /*Kokkos::parallel_for("inner_loop",inner_policy, KOKKOS_LAMBDA(const int j){
                usleep(1000);
            }
            );*/
            //j=i, j<N (M=64,N=64,usleep=1000)
            for(int j=i; j<N; j++){
                usleep(2*1e6);
                
            }
        });

}

int main(int argc, char *argv[]){
    
    tuned_kernel(
        argc, argv,
        [&](const int total_iters) {

            std::vector<std::string> candidates = {"dynamic_schedule","static_schedule","ghost_schedule"};
        

            //Output variable - schedule
            Kokkos::Tools::Experimental::VariableInfo schedule_out_info;
            schedule_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_string;
            schedule_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            schedule_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            schedule_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());

            size_t schedule_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tuning_pplayground.schedule_out",schedule_out_info);

            return std::make_tuple(schedule_out_value_id);
        },
        [&] (const int iter, const int total_iters, size_t schedule_out_value_id){

            //The second argument to make_varaible_value might be a default value
            std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
                Kokkos::Tools::Experimental::make_variable_value(schedule_out_value_id,"dynamic_schedule")
            };

            size_t context = Kokkos::Tools::Experimental::get_new_context_id();
            Kokkos::Tools::Experimental::begin_context(context);
            Kokkos::Tools::Experimental::request_output_values(context, 1, answer_vector.data());

            std::string scheduleType = answer_vector[0].value.string_value;
          
            if(scheduleType.compare("dynamic_schedule")==0){
                doParallel<Kokkos::Schedule<Kokkos::Dynamic>>();
            }
            else{
                doParallel<Kokkos::Schedule<Kokkos::Static>>();
            }

            /*Kokkos::fence();
            double time_elapsed = timer.seconds();
            cout<<"Execution Time: "<<time_elapsed*1e6<<" microseconds"<<endl;*/

            Kokkos::Tools::Experimental::end_context(context);
        } 
    );
}