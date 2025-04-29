
echo "resources/cpu_percent"
python visualizer.py -m "resources/cpu_percent" $1

echo "resources/gpu_memory_percentage"
python visualizer.py -m "resources/gpu_memory_percentage" $1

echo "resources/gpu_memory_used"
python visualizer.py -m "resources/gpu_memory_used" $1

echo "resources/gpu_power"
python visualizer.py -m "resources/gpu_power" $1

echo "resources/gpu_utilization"
python visualizer.py -m "resources/gpu_utilization" $1

echo "resources/ram_percent"
python visualizer.py -m "resources/ram_percent" $1
