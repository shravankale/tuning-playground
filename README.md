
## OpenMP Benchmarks: Tuning OpenMP examples with Apex Tuner

The *examples* directory consists of sub-directories for individual OpenMP benchmarks. 

Each of the sub-directories consists of the following files:
1. *example name*_*tuning variable*.sh (Run Script)
    - Consists of hardcoded paths for the apex executable and openmp example executable that need to be changed accordingly

2. *sourceme.sh*
    - Consists of environment variables and its preselcted varaibles required for the tuning task.

**Note:** A separate *cleanup.sh* script is available under the examples folder to clear temporary experiment files across runs.