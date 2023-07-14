
OpenMP Examples: Tuning OpenMP examples with Apex Tuner

The examples directory consists of sub-directories for individual OpenMP example. 
Each of the sub-directories consists of the following files:
1. *example name*_*tuning variable*.sh //Run script
    Consists of hardcoded paths for the apex executable and openmp example executable that need to be changed accordingly

3. sourceme.sh //source sourceme.sh
    Consists of environment variables and its preselcted varaibles required for the tuning task.

Note: A seperate cleanup.sh script is available under the examples folder to clear temporary experiment files across runs. 
