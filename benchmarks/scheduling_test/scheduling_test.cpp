
/*
    'tuning_playground' plugin for tuning the schedule for a matrix-multiplcation kerenel

    Values for the above varaibles:
    1. schedule (fixed) = Kokkos provides support only for dynamic and static scheduling

    Dimensions of matrices: M,N,P
*/

//Loops for trainglular solve. Outter loop

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

const int M=1e5;

using vector = Kokkos::View<double *, Kokkos::OpenMP::memory_space>;
int lb=100, ub=999;


template <typename T>
void doParallel(vector &re, vector &ar1, vector &ar2){

        
        Kokkos::parallel_for(re.size(), KOKKOS_LAMBDA(const int i) {
            double sleepDuration = static_cast<double>(i) / 1e6;
            //switch to usleep
            sleep(sleepDuration);
        });
        

}

int main(int argc, char *argv[]){
    
    //using KTE = Kokkos::Tools::Experimental;

    tuned_kernel(
        argc, argv,
        [&](const int total_iters) {

            /*
                Drop the input_values semantics in the setup, and the tunable.
                The current setup in nthreads/tiling is essentially tuning y|y
                which means nothing.  
            */
            
            srand(time(0));

            /* Declare/Init re,ar1,ar2 */
            vector ar1("array1",M), ar2("array2",M), re("Result",M);

            for(int i=0; i<M; i++){
                
                ar1(i)=(rand() % (ub - lb + 1)) + lb;
                ar2(i)=(rand() % (ub - lb + 1)) + lb;
                re(i)=0;
            }
                

            std::vector<std::string> candidates = {"dynamic_schedule","static_schedule","ghost_schedule"};
        

            //Output variable - schedule
            Kokkos::Tools::Experimental::VariableInfo schedule_out_info;
            schedule_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_string;
            schedule_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            //Kale - Unboudned or set?
            schedule_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            //Check if second argument is correct
            schedule_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());

            size_t schedule_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tuning_pplayground.schedule_out",schedule_out_info);

            return std::make_tuple(schedule_out_value_id, re, ar1, ar2);
        },
        [&] (const int iter, const int total_iters, size_t schedule_out_value_id,  vector re, vector ar1, vector ar2){

            //The second argument to make_varaible_value might be a default value
            std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
                Kokkos::Tools::Experimental::make_variable_value(schedule_out_value_id,"dynamic_schedule")
            };

            size_t context = Kokkos::Tools::Experimental::get_new_context_id();
            Kokkos::Tools::Experimental::begin_context(context);
            Kokkos::Tools::Experimental::request_output_values(context, 1, answer_vector.data());


            //Scheduling
            //answer_vector[0].value.int_value;
            std::string scheduleType = answer_vector[0].value.string_value;
            /*
            std::cout<<"X -  scheduleType: "<<scheduleType<<std::endl;
            std::cout<<"Looping through answer vector: "<<std::endl;
            for(auto &i: answer_vector){
                std::cout<<i.value.string_value;
            }
            std::cout<<std::endl;
            */


            if(scheduleType.compare("dynamic_schedule")==0){
                //std::cout<<"X -  Tunning: "<<scheduleType<<std::endl;
                doParallel<Kokkos::Schedule<Kokkos::Dynamic>>(re,ar1,ar2);
            }
            else{
                //Static scheduling
                //std::cout<<"X -  Tunning: "<<scheduleType<<std::endl;
                doParallel<Kokkos::Schedule<Kokkos::Static>>(re,ar1,ar2);
            }

        
            /*
            if(scheduleType.compare("dynamic_schedule")){
                //Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::Rank<3>> policy({0,0,0},{M,N,P});

                auto policy = getMDRangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>();
            }
            else{
                //Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Static>, Kokkos::Rank<3>> policy({0,0,0},{M,N,P});
                auto policy = getMDRangePolicy<Kokkos::Schedule<Kokkos::Static>>();
            }

            

            //Iteration Range
            //Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<3>> policy({0,0,0},{M,N,P});
            //Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::OpenMP, Kokkos::Schedule<static|dynamic|...>>, policy({0,0,0},{M,N,P},{tile_M,tile_N,tile_P})
            
            //Iteration indices (i,j,k) here are mapped to cores and the cores executes the computational body for the given indicies. 
            Kokkos::parallel_for(
                "mm2D", policy, KOKKOS_LAMBDA(int i, int j, int k){
                    re(i,j) += ar1(i,j) * ar2(j,k);
                }
            );
            */

            /*Kokkos::fence();
            double time_elapsed = timer.seconds();
            cout<<"Execution Time: "<<time_elapsed*1e6<<" microseconds"<<endl;*/

            Kokkos::Tools::Experimental::end_context(context);
        } 
    );
}