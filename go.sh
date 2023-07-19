
NVC_PATH="/home/users/shravank/libs/nvhpc/Linux_x86_64/22.7/compilers/bin/nvc"
NVC_PP_PATH="/home/users/shravank/libs/nvhpc/Linux_x86_64/22.7/compilers/bin/nvc++"
BFD_PATH="/usr/bin"
export NVHPC_CUDA_HOME="/home/users/shravank/libs/nvhpc/Linux_x86_64/22.7/cuda/11.0"
CUDA_PATH="/home/users/shravank/libs/nvhpc/Linux_x86_64/22.7/cuda/11.0"


cwd=`pwd`
srcdir=${cwd}/src
builddir=${cwd}/build
instdir=${cwd}/install

mkdir -p ${srcdir}
mkdir -p ${builddir}
mkdir -p ${instdir}

getsrc() {
    # Get source code
    cd ${srcdir}
    if [ ! -d tuning-playground ] ; then
        git clone https://github.com/DavidPoliakoff/tuning-playground.git
    fi

    cd ${srcdir}
    if [ ! -d kokkos-kernels ] ; then
        git clone https://github.com/kokkos/kokkos-kernels.git
    fi

    cd ${srcdir}
    if [ ! -d kokkos ] ; then
        git clone https://github.com/kokkos/kokkos.git
        cd ${srcdir}/kokkos
        git checkout develop
    fi

    cd ${srcdir}
    if [ ! -d apex ] ; then
        # Better, fork this repo on github and clone your fork!
        git clone https://github.com/shravankale/apex.git
        cd ${srcdir}/apex
        git checkout develop
    fi

    cd ${cwd}
}

buildkokkos() {
    # Build Kokkos - expect warnings from the compiler
    rm -rf ${builddir}/kokkos
    mkdir ${builddir}/kokkos
    cd ${builddir}/kokkos

    /bin/cmake \
    ${srcdir}/kokkos \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=${NVC_PP_PATH} \
    -DCMAKE_INSTALL_PREFIX=${instdir}/kokkos \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_TUNING=ON \
    -DKokkos_ENABLE_HWLOC=ON \
    -DKokkos_ENABLE_TESTS=OFF \
    -DKokkos_ARCH_NATIVE=ON 
    
    #-DKokkos_ENABLE_CUDA=ON \
    #-DCUDA_ROOT=${CUDA_PATH} \
    #-DKokkos_ARCH_PASCAL60=ON

    
    make -j
    make install
}

buildplayground() {
    # Build tuning-playground

    rm -rf ${builddir}/tuning-playground
    mkdir ${builddir}/tuning-playground
    cd ${builddir}/tuning-playground

    /bin/cmake ${srcdir}/tuning-playground \
    -DCMAKE_CXX_COMPILER=${NVC_PP_PATH} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=${instdir}/tuning-playground \
    -DKokkos_ROOT=${instdir}/kokkos/lib/cmake/Kokkos \
    -DCMAKE_CXX_EXTENSIONS=OFF
    make -j
    make install
}

buildkernels() {
    rm -rf ${builddir}/kokkos-kernels
    mkdir ${builddir}/kokkos-kernels
    cd ${builddir}/kokkos-kernels

    set -x
    cmake ${srcdir}/kokkos-kernels \
    -DCMAKE_CXX_COMPILER=`which nvc++` \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=${instdir}/kokkos-kernels \
    -DKokkos_ROOT=${instdir}/kokkos \
    -DKokkosKernels_ENABLE_TPL_CUBLAS=Off \
    -DKokkosKernels_ENABLE_TPL_CUSPARSE=Off \
    -DKokkosKernels_ENABLE_TESTS=ON
    set +x

    make -j32 -l40 sparse_spmv
}

# Build APEX
buildapex() {
    rm -rf ${builddir}/apex
    mkdir ${builddir}/apex
    cd ${builddir}/apex

    set -x
    /bin/cmake ${srcdir}/apex \
    -DCMAKE_C_COMPILER=${NVC_PATH} \
    -DCMAKE_CXX_COMPILER=${NVC_PP_PATH} \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DAPEX_DEBUG=FALSE \
    -DCMAKE_INSTALL_PREFIX=${instdir}/apex \
    -DAPEX_BUILD_TESTS=FALSE \
    -DAPEX_WITH_BFD=TRUE \
    -DAPEX_WITH_CUDA=FALSE \
    -DAPEX_WITH_PLUGINS=TRUE \
    -DAPEX_WITH_ACTIVEHARMONY=FALSE \
    -DAPEX_WITH_OMPT=TRUE \
    -DAPEX_BUILD_OMPT=FALSE \
    -DBFD_ROOT=${BFD_PATH}

    #-DCUDA_ROOT=${CUDA_PATH} \
    #-DCUDAToolkit_ROOT=${CUDA_PATH} \
    #-DCMAKE_CUDA_ARCHITECTURES=60 \
    

    make -j -l32
    make install
}

##getsrc
##buildkokkos
buildplayground
#buildkernels     # <-- won't compile. looking into it.
##buildapex

