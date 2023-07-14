
/*
    'tuning_playground' plugin for tuning number of openmp threads, schedule, and tilesizes in a matrix-multiplcation kerenel

    Values for the above varaibles:
    1. openmp_threads(varaible) = The nproc varaible defines the maxiumum amount of threads. Modify according to hardware
    2. schedule (fixed) = Kokkos provides support only for dynamic and static scheduling
    3. tilesizes (fixed) = Factors of matrix dimension size.

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
#include <stdexcept>

const int M=64;
const int N=64;
const int P=64;

using matrix2d = Kokkos::View<int **, Kokkos::OpenMP::memory_space>;
int lb=100, ub=999;

#define min(x,y)    x < y ? x : y

std::vector<int64_t> factorsOf(const int &size){

    std::vector<int64_t> factors;
    for(int i=1; i<size; i++){
        if(size % i == 0){
            factors.push_back(i);
        }
    }
    //Inserting a value that's a repeat of the last element - Remove after Apex fix
    factors.push_back(factors.back()+int((factors.back()*0.5)));
    //factors.erase(factors.begin());

    return factors;
}

std::vector<int> getTiledIterations(const int &init_val, const int &max_val, const int &tile_size){

    std::vector<int> iters;
    for(int i=init_val; i<max_val; i+=tile_size){
        iters.push_back(i);
    }

    if(iters.empty()){
        throw std::runtime_error(std::string("Error: iters is empty "));
    }
    
    return iters;
}

void printFactors(std::vector<int64_t> &candidates, const char &X, const int &Y){
    std::cout<<"Tiling options for "<<X<<": "<<Y<<std::endl;
            for(auto &i : candidates){ std::cout<<i<<", "; }
            std::cout<<std::endl;
}

template <typename executionType, typename scheduleType>
void doParallelNew(
    matrix2d &re, matrix2d &ar1, matrix2d &ar2,
    const int &set_num_threads,
    const int &ti, const int &tj, const int &tk){

    //Setting num threads using hiearchial parallelism
    
    using TeamHandle = typename Kokkos::TeamPolicy<executionType, scheduleType>::member_type;
    int league_size = 1;
    int team_size = set_num_threads;

    Kokkos::TeamPolicy<executionType, scheduleType> policy(league_size, team_size);

    /*
    //Problems: 
    //Thread Print Pattern: Not discernable
    //Takes excessive time over below solutions
    Kokkos::parallel_for(
        "mm-all", policy, KOKKOS_LAMBDA(TeamHandle const& team){

            printf("fTeam_size: %d \n",team.team_size());

            Kokkos::MDRangePolicy<executionType, Kokkos::Rank<3>> policy_ijk({0,0,0},{M,N,P},{ti,tj,tk});

            Kokkos::parallel_for("loop_tiling", policy_ijk, KOKKOS_LAMBDA(int i, int j, int k){

                printf("fTeam_rank: %d \n",team.team_rank());

                re(i,k) += ar1(i,j) * ar2(j,k);
            });
        }
    );
    */

    

    
    //Using Kokkos::TeamThreadRange
    //Thread Print Patern: As expected
    Kokkos::parallel_for(
        "mm-all-TeamThreadMDRange", policy, KOKKOS_LAMBDA(TeamHandle const& team){

            auto teamThreadRange = Kokkos::TeamThreadRange(team,1);

            printf("fTeam_size: %d \n",team.team_size());

            Kokkos::parallel_for(teamThreadRange, KOKKOS_LAMBDA(int a){

                std::cout<<"team_rank: "<<team.team_rank()<<std::endl;
                printf("fTeam_rank: %d \n",team.team_rank());

                Kokkos::MDRangePolicy<executionType,Kokkos::Rank<3>> policy_ijk({0,0,0},{M,N,P},{ti,tj,tk});
                Kokkos::parallel_for("loop_tiling", policy_ijk, KOKKOS_LAMBDA(int i, int j, int k){
                    re(i,k) += ar1(i,j) * ar2(j,k);
                });

            });

            
                

        }
    );
    
    
    /*
    //Another way to use teamthreadmdrange - Tiles are threaded seperately
    //Problem: Any tile_size value of 1 leads to an std::invalid_argument
    //Deduction: Removing tile_size 1 as an option leads to successful program execution
    //Fix: Unknown
    //Thread Pattern: As exepected
    std::vector<int> fti = getTiledIterations(1,M,ti); // (0,10,2)= [0,2,4,6,8]
    std::vector<int> ftj = getTiledIterations(1,N,tj);
    std::vector<int> ftk = getTiledIterations(1,P,tk);

    Kokkos::parallel_for(
        "mm-all-TeamThreadMDRange", policy, KOKKOS_LAMBDA(TeamHandle const& team){

            //printf("fTeam_size: %d \n",team.team_size());

            auto teamThreadMDRange = Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team, fti.size(),ftj.size(),ftk.size());

            Kokkos::parallel_for(teamThreadMDRange, KOKKOS_LAMBDA(int ni, int nj, int nk){

                //printf("fTeam_rank: %d \n",team.team_rank());

                int ii = fti[ni];
                int jj = ftj[nj];
                int kk = ftk[nk];

                Kokkos::MDRangePolicy<executionType,Kokkos::Rank<3>> policy_ijk(
                    {ii,jj,kk},
                    {min(ii+ti,M),min(jj+tj,N),min(kk+tk,P)}
                );
                Kokkos::parallel_for("loop_tiling", policy_ijk, KOKKOS_LAMBDA(int i, int j, int k){
                    re(i,k) += ar1(i,j) * ar2(j,k);
                });

            });   
        }
    );
    */
    
   

}

int main(int argc, char *argv[]){
    
    //using KTE = Kokkos::Tools::Experimental;

    tuned_kernel(
        argc, argv,
        [&](const int total_iters) {

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


            //Output variable - schedule
            std::vector<std::string> candidates_schedule = {"dynamic_schedule","static_schedule", "ghost_schedule"};

            Kokkos::Tools::Experimental::VariableInfo schedule_out_info;
            schedule_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_string;
            schedule_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            schedule_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            schedule_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());

            size_t schedule_out_value_id = Kokkos::Tools::Experimental::declare_output_type("schedule_out",schedule_out_info);

            //Output Variable - nthreads
            int nproc = 50;
            std::vector<int64_t> candidates_nthreads;
            for(int64_t i=1; i<=nproc; i++){
                candidates_nthreads.push_back(i);
            }

            Kokkos::Tools::Experimental::VariableInfo nthreads_out_info;
            nthreads_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
            nthreads_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            nthreads_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            nthreads_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_nthreads.size(),candidates_nthreads.data());

            size_t nthreads_out_value_id = Kokkos::Tools::Experimental::declare_output_type("nthreads_out",nthreads_out_info);

            //Output Variables - tilling
            std::vector<int64_t> candidates_ti=factorsOf(M), candidates_tj=factorsOf(N), candidates_tk=factorsOf(P);
            printFactors(candidates_ti, 'M',M); printFactors(candidates_tj, 'N',N); printFactors(candidates_tk, 'P',P);

            Kokkos::Tools::Experimental::VariableInfo ti_out_info, tj_out_info, tk_out_info;
            ti_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
            tj_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;;
            tk_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;;

            ti_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            tj_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
            tk_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;

            ti_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            tj_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
            tk_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;

            ti_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_ti.size(),candidates_ti.data());
            tj_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_tj.size(),candidates_tj.data());
            tk_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_tk.size(),candidates_tk.data());

            size_t ti_out_value_id = Kokkos::Tools::Experimental::declare_output_type("ti_out",ti_out_info);
            size_t tj_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tj_out",tj_out_info);
            size_t tk_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tk_out",tk_out_info); 


            


            return std::make_tuple(
                schedule_out_value_id,nthreads_out_value_id, 
                ti_out_value_id,tj_out_value_id,tk_out_value_id,
                re, ar1, ar2);
        },
        [&] (
            const int iter, const int total_iters, 
            size_t schedule_out_value_id, size_t nthreads_out_value_id,
            size_t ti_out_value_id, size_t tj_out_value_id, size_t tk_out_value_id, 
            matrix2d re, matrix2d ar1, matrix2d ar2){

            //The second argument to make_varaible_value might be a default value
            std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
                Kokkos::Tools::Experimental::make_variable_value(schedule_out_value_id,"static_schedule"),
                Kokkos::Tools::Experimental::make_variable_value(nthreads_out_value_id,int64_t(1)),
                Kokkos::Tools::Experimental::make_variable_value(ti_out_value_id, int64_t(1)),
                Kokkos::Tools::Experimental::make_variable_value(tj_out_value_id, int64_t(1)),
                Kokkos::Tools::Experimental::make_variable_value(tk_out_value_id, int64_t(1))
            };

            size_t context = Kokkos::Tools::Experimental::get_new_context_id();
            Kokkos::Tools::Experimental::begin_context(context);
            Kokkos::Tools::Experimental::request_output_values(context, 5, answer_vector.data());


            //Get Tuning values
            std::string scheduleType = answer_vector[0].value.string_value;
            int set_num_threads = answer_vector[1].value.int_value;
            int ti = answer_vector[2].value.int_value;
            int tj = answer_vector[3].value.int_value;
            int tk = answer_vector[4].value.int_value;

            std::cout<<"scheduleType: "<<scheduleType<<std::endl;
            std::cout<<"set_num_threads: "<<set_num_threads<<std::endl;
            std::cout<<"ti: "<<ti<<std::endl;
            std::cout<<"tj: "<<tj<<std::endl;
            std::cout<<"tk: "<<tk<<std::endl;
                       
            
            int res = scheduleType.compare("static_schedule");
            if(res==0){
                std::cout<<"Doing STATIC"<<std::endl;
                doParallelNew<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Static>>(re,ar1,ar2,set_num_threads, ti, tj, tk);
            }
            else{
                std::cout<<"Doing DYNAMIC"<<std::endl;
                doParallelNew<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Dynamic>>(re,ar1,ar2,set_num_threads, ti, tj, tk);
            }
            /*
            if(scheduleType.compare("dynamic_schedule")==0){
                //std::cout<<"X -  Tunning: "<<scheduleType<<std::endl;
                doParallel<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Dynamic>>(re,ar1,ar2,set_num_threads, ti, tj, tk);
            }
            else{
                //Static scheduling
                //std::cout<<"X -  Tunning: "<<scheduleType<<std::endl;
                doParallel<Kokkos::OpenMP,Kokkos::Schedule<Kokkos::Static>>(re,ar1,ar2,set_num_threads, ti, tj, tk);
            }
            */

            /*Kokkos::fence();
            double time_elapsed = timer.seconds();
            cout<<"Execution Time: "<<time_elapsed*1e6<<" microseconds"<<endl;*/

            Kokkos::Tools::Experimental::end_context(context);
        } 
    );
}