conda activate arsq_rlb

export CQN_PATH=$(pwd)
export COPPELIASIM_ROOT=${CQN_PATH}/thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
