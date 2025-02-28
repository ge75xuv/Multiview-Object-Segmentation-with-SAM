python clean_gpu_garbage.py

echo "run nvidia-smi"
nvidia-smi
echo "\n"
echo "run nvidia-smi with User ID"
ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`