mkdir -p gpu_results
cd scripts/weak_scaling || exit
echo "Running weak scaling benchmark"
python weak_scaling_gpu.py
mv weak_scaling_gpu.csv ../../gpu_results/
cd ../strong_scaling || exit
echo "Running strong scaling benchmark"
python strong_scaling_gpu.py
mv strong_scaling_gpu.csv ../../gpu_results/
cd ../higgs || exit
if [[ ! -f "./data/higgs.pq" ]]; then
    echo "Downloading Higgs dataset"
    sh ./download_dataset.sh
fi
echo "Running Higgs benchmark"
python higgs_gpu.py
mv higgs_gpu.csv ../../gpu_results/
