
/*
    'tuning_playground' plugin for tuning number of openmp threads in a matrix-multiplcation kerenel

    Values for the above varaibles:
    1. openmp_threads(varaible) = The nproc varaible defines the maxiumum amount of threads. Modify according to hardware
  
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

const int M=512;
const int N=512;
const int P=512;

using matrix2d = Kokkos::View<int **, Kokkos::OpenMP::memory_space>;
int lb=100, ub=999;

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
            size_t nthreads_out_value_id;
            int nproc = 50;
            std::vector<int64_t> candidates;
            for(int64_t i=1; i<=nproc; i++){
                candidates.push_back(i);
            }

            //Input Varaible - nthreads
            ////Kokkos::Tools::Experimental::VariableInfo nthreads_inp_info;
            ////nthreads_inp_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
            ////nthreads_inp_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ordinal;
            ////nthreads_inp_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            //Probably need to new (malloc and memcpy) candidates vector
            ////nthreads_inp_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());

            //Output variable - nthreads
            Kokkos::Tools::Experimental::VariableInfo nthreads_out_info;
            nthreads_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
            nthreads_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            nthreads_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            //Check if second argument is correct
            nthreads_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),candidates.data());

            ////nthreads_inp_value_id = Kokkos::Tools::Experimental::declare_input_type("tuning_playground.nthreads_inp",nthreads_inp_info);
            nthreads_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tuning_pplayground.nthreads_out",nthreads_out_info);

            //Need to return views too
            ////return std::make_tuple(nthreads_inp_value_id,nthreads_out_value_id, re, ar1, ar2);
            return std::make_tuple(nthreads_out_value_id, re, ar1, ar2);
        },
        ////[&] (const int iter, const int total_iters, size_t nthreads_inp_value_id, size_t nthreads_out_value_id,  matrix2d re, matrix2d ar1, matrix2d ar2){
        [&] (const int iter, const int total_iters, size_t nthreads_out_value_id,  matrix2d re, matrix2d ar1, matrix2d ar2){
            //Not sure the kind of value nthreads_inp should have
            ////int64_t nthreads_inp = iter%10;

            ////std::vector<Kokkos::Tools::Experimental::VariableValue> feature_vector{
                ////Kokkos::Tools::Experimental::make_variable_value(nthreads_inp_value_id,nthreads_inp)
            ////};
            //The second argument to make_varaible_value might be a default value
            std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
                Kokkos::Tools::Experimental::make_variable_value(nthreads_out_value_id,int64_t(1))
            };

            size_t context = Kokkos::Tools::Experimental::get_new_context_id();
            Kokkos::Tools::Experimental::begin_context(context);
            ////Kokkos::Tools::Experimental::set_input_values(context, 1, feature_vector.data());
            Kokkos::Tools::Experimental::request_output_values(context, 1, answer_vector.data());

            
            //Get this set_num_threads value from request output values
            //omp_set_dynamic?
            int set_num_threads = answer_vector[0].value.int_value;
            //std::cout<<"Set OMP Num Threads: "<<set_num_threads<<std::endl;
            printf("Set OMP Num Threads: %d\n",set_num_threads);

            //Attempting to set num threads by using hiearchial parallelism

            using TeamHandle = Kokkos::TeamPolicy<Kokkos::OpenMP>::member_type;

            int league_size = 1;
            int team_size = set_num_threads; 
            Kokkos::TeamPolicy<Kokkos::OpenMP> policy(league_size,team_size);

            Kokkos::parallel_for(
                "mm-nthread", policy, KOKKOS_LAMBDA(TeamHandle const& team){

                    //int leagueRank = team.league_rank();
                    //printf("LeagueRank: %d",leagueRank);

                    //Replace the auto
                    auto teamThreadMDRange = Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team, M, N, P);

                    Kokkos::parallel_for(teamThreadMDRange, KOKKOS_LAMBDA(int i, int j, int k){

                        //Get Team Thread Ids 
                        //int teamRank = team.team_rank();
                        //int teamSize = team.team_size();

                        //printf("Tid: %d/%d",teamRank,teamSize);

                        re(i,j) += ar1(i,j) * ar2(j,k);
                    });

                }
            );



            /*
            Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<3>> policy({0,0,0},{M,N,P});
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
            printf("End Tuning Context");
            Kokkos::Tools::Experimental::end_context(context);
        } 
    );
}