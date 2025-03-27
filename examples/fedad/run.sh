# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5
# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 69 -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5

# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 42 -b 128  --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 5e-4
# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 42 -b 256 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 5e-4
# python main.py fedours -e 250 -k 5 -t 100 --distill kl --seed 42 -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 5e-4

# echo "ALPHA => 0.5"
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 420 -b 64  --base ./models/five-resnet-alpha-0_5/  --alpha 0.5 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 69  -b 64 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 42  -b 64 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
#
# echo "ALPHA => 0.1"
# python main.py fedavg -e 5 -k 5 -t 40  --distill kl --seed 420 -b 64 --base ./models/remote-01/  --alpha 0.1 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40  --distill kl --seed 69  -b 64 --base ./models/remote-01/ --alpha 0.1 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40  --distill kl --seed 42  -b 64 --base ./models/remote-01/ --alpha 0.1 --lr 1e-3
#
# echo "ALPHA => 0.005"
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 420 -b 64 --base ./models/remote-005/  --alpha 0.05 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 69  -b 64 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 42  -b 64 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
#
# echo "ALPHA => 0.001"
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 420 -b 64 --base ./models/remote-001/  --alpha 0.01 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 69  -b 64 --base ./models/remote-001/ --alpha 0.01 --lr 1e-3
# python main.py fedavg -e 5 -k 5 -t 40 --distill kl --seed 42  -b 64 --base ./models/remote-001/ --alpha 0.01 --lr 1e-3

# ---

# python main.py tinyours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001/  --alpha 0.01 --lr 1e-1
# python main.py tinyours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001/  --alpha 0.01 --lr 1e-3
# python main.py tinyours -e 250 -k 5 -t 30 --distill kl --seed 420 -b 512 --base ./models/remote-001/  --alpha 0.01 --lr 0.001

# ---

python main.py fedours -e 250 -k 5 -t 15 --distill kl --seed 420 -b 128 --base ./models/remote-001 --alpha 0.01 --lr 1e-4
# python main.py fedours -e 250 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 5e-4
# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 64 --base ./models/remote-001 --alpha 0.01 --lr 1e-3

# python main.py fedours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 1e-3
# python main.py tinyours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 1e-3
#
# python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 8e-4
#python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 8e-4
# python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-001 --alpha 0.01 --lr 1e-3


# python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py fedours -e 250 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3

# python main.py fedours  -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-005 --alpha 0.05 --lr 1e-3
# python main.py tinyours -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-005 --alpha 0.05 --lr 1e-3
# python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-005 --alpha 0.05 --lr 1e-3
# python main.py lmd -e 250 -k 5 -t 10 --distill kl --seed 420 -b 512 --base ./models/remote-005 --alpha 0.05 --lr 1e-3

# ---

# python main.py train -e 200 -k 5 -b 16 -t 20 --alpha 0.05 --base ./models/remote-005 --distill kl --lr 0.001 --seed 42
# python main.py train -e 200 -k 5 -b 16 -t 20 --alpha 0.1  --base ./models/remote-01  --distill kl --lr 0.001 --seed 42

# --- lmd
# echo "ALPHA => 0.5"
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/  --alpha 0.5 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3

# echo "ALPHA => 0.1"
# python main.py lmd -e 5 -k 5 -t 70  --distill kl --seed 420 -b 512 --base ./models/remote-01/  --alpha 0.1 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70  --distill kl --seed 69  -b 512 --base ./models/remote-01/ --alpha 0.1 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70  --distill kl --seed 42  -b 512 --base ./models/remote-01/ --alpha 0.1 --lr 1e-3

# echo "ALPHA => 0.005"
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-005/  --alpha 0.05 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3

# echo "ALPHA => 0.001"
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-001/  --alpha 0.01 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/remote-001/ --alpha 0.01 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/remote-001/ --alpha 0.01 --lr 1e-3

# expours
#
# echo "ALPHA => 0.5"
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/  --alpha 0.5 --lr 1e-3
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3


# echo "ALPHA => 0.005"
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-005/  --alpha 0.05 --lr 1e-3
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
# python main.py expours -e 10 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3

# echo "ALPHA => 0.5"
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/  --alpha 0.5 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3


# echo "ALPHA => 0.005"
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-005/  --alpha 0.05 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
# python main.py lmd -e 5 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3

# combined

# echo "ALPHA => 0.5"
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/five-resnet-alpha-0_5/  --alpha 0.5 --lr 1e-3
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/five-resnet-alpha-0_5/ --alpha 0.5 --lr 1e-3


# echo "ALPHA => 0.005"
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 420 -b 512 --base ./models/remote-005/  --alpha 0.05 --lr 1e-3
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 69  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
# python main.py combined -e 10 -k 5 -t 70 --distill kl --seed 42  -b 512 --base ./models/remote-005/ --alpha 0.05 --lr 1e-3
