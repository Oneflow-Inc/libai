set -u

DATASET_MOUNT_DIR="/dataset/cf4e4dae/v1/"
DATASET_SAVE_DIR="/home/StanfordCars"

# Install tmux
apt-get update
apt-get install -y tmux

# Install LiBai
cd /workspace
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
pip install pybind11
pip install -e .

# Prepare StanfordCars dataset
cd $DATASET_MOUNT_DIR
mkdir $DATASET_SAVE_DIR
tar -zxf car_ims.tgz -C $DATASET_SAVE_DIR
cp cars_annos.mat $DATASET_SAVE_DIR
cp cars_test_annos_withlabels.mat $DATASET_SAVE_DIR

python ./standford_cars_preprocess.py --dataset_root $DATASET_SAVE_DIR

# Install scipy, wiich is used when reading StanfordCars dataset's annnotation file
pip install scipy
