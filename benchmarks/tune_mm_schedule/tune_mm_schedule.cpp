
/*
    'tuning_playground' plugin for tuning the schedule for a matrix-multiplcation kerenel

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

const int M=64;
const int N=64;
const int P=64;

using matrix2d = Kokkos::View<int **, Kokkos::OpenMP::memory_space>;
int lb=100, ub=999;

/*
// Schedules for Execution Policies
struct Static {};
struct Dynamic {};

// Schedule Wrapper Type
template <class T>
struct Schedule {
  static_assert(std::is_same<T, Static>::value ||
                    std::is_same<T, Dynamic>::value,
                "Kokkos: Invalid Schedule<> type.");
  using schedule_type = Schedule;
  using type          = T;
};
*/

/*enum Sched
{
    s = Kokkos::Schedule<Kokkos::Static>,
    d = Kokkos::Schedule<Kokkos::Dynamic>
};*/

/*struct Sched{
    using static_schedule = Kokkos::Schedule<Kokkos::Static>;
    using dynamic_schule = Kokkos::Schedule<Kokkos::Dynamic>;
};
*/

/*template <typename T>
Kokkos::MDRangePolicy<Kokkos::OpenMP, T, Kokkos::Rank<3>> getMDRangePolicy(){

        Kokkos::MDRangePolicy<Kokkos::OpenMP, T, Kokkos::Rank<3>> policy({0,0,0},{M,N,P});

        return policy;
}*/

template <typename T>
void doParallel(matrix2d &re, matrix2d &ar1, matrix2d &ar2){

        
        Kokkos::MDRangePolicy<Kokkos::OpenMP, T, Kokkos::Rank<3>> policy({0,0,0},{M,N,P});

        Kokkos::parallel_for(
                "mm2D", policy, KOKKOS_LAMBDA(int i, int j, int k){
                    re(i,j) += ar1(i,j) * ar2(j,k);
                }
            );
        
        /* 
        Kokkos::RangePolicy<Kokkos::OpenMP, T> policy(0,M);
        Kokkos::parallel_for("mm2D", policy, 
                    KOKKOS_LAMBDA(int i){
                        sleep(0.1*i);
                    }
                );
        */

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
            matrix2d ar1("array1",M,N), ar2("array2",N,P), re("Result",M,P);

            for(int i=0; i<M; i++){
                for(int j=0; j<N; j++){
                        ar1(i,j)=(rand() % (ub - lb + 1)) + lb;
                }
            }
                
            for(int j=0; j<N; j++){
                for(int k=0; k<P; k++){
                        ar2(j,k)=(rand() % (ub - lb + 1)) + lb; 
                }
            }

            for(int i=0; i<M; i++){
                for(int k=0; k<P; k++){
                        re(i,k)=0;
                }
            }



            ////size_t nthreads_inp_value_id;
            //size_t nthreads_out_value_id;
            //std::vector<int64_t> candidates{1,2,3,4,5,6,7,8,9,10};
            //std::vector<Kokkos::Schedule<>> candidates{Kokkos::Schedule<Dynamic>, Kokkos::Schedule<Static>}
            /*std::vector<Sched> candi;
            Sched static_schedule = s;
            Sched dynamic_schedule = d;

            candi.push_back(static_schedule);
            candi.push_back(dynamic_schedule);
            
            //std::vector<std::variant<Kokkos::Schedule<Static>,Kokkos::Schedule<Dynamic>>> ;
            std::vector<Sched> cd;
            cd.push_back(Sched::static_schedule);
            cd.push_back(Sched::dynamic_schule); 
            */

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
        [&] (const int iter, const int total_iters, size_t schedule_out_value_id,  matrix2d re, matrix2d ar1, matrix2d ar2){

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