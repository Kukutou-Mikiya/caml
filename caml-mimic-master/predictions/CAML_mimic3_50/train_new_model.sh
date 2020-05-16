python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 conv_attn 200 --filter-size 10 --num-filter-maps 50 --dropout 0.2 --patience 10 --criterion prec_at_5 --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
#跑40个epoch，卷积核数量翻4倍，性能有明显改进
python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 conv_attn 200 --filter-size 10 --num-filter-maps 500 --dropout 0.2 --patience 10  --criterion f1_macro --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
#测试用
python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 my_conv_attn 200 --filter-size 10 --num-filter-maps 300 --dropout 0.2 --patience 100  --criterion f1_macro --lr 0.003 --batch-size 8 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
#普通CNN
python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 cnn_vanilla 1 --filter-size 10 --num-filter-maps 400 --dropout 0.2 --patience 10  --criterion f1_macro --lr 0.0001 --batch-size 8 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu