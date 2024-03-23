# CrossEntropyLoss

# Train CrossEntropyLoss
python main.py \
--lr 0.1 \
--loss CrossEntropyLoss \
--save_path test/CrossEntropyLoss \
--mode train \
--max_num_epochs 200 \
--dataset "CIFAR100"

# Test best CrossEntropyLoss model
python test.py \
--lr 0.1 \
--save_path test/CrossEntropyLoss \
--model_path test/CrossEntropyLoss/best.pth \
--loss "CrossEntropyLoss" \
--dataset "CIFAR100"

# Test last CrossEntropyLoss model
python test.py \
--lr 0.1 \
--save_path test/CrossEntropyLoss \
--model_path test/CrossEntropyLoss/last.pth \
--loss "CrossEntropyLoss" \
--dataset "CIFAR100"