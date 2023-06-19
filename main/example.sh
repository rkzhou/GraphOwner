python benign.py --use_org_node_attr --train_verbose

nohup python -u attack.py --use_org_node_attr --train_verbose --target_class 0 --train_epochs 20 > ../attack.log 2>&1 &

python3 benign.py --use_org_node_attr --train_verbose --dataset='Cora' --model='sage'

python3 attack.py --use_org_node_attr --train_verbose --dataset='Cora' --model='sage' --target_class=1