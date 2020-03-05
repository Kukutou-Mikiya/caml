python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 conv_attn 200 --filter-size 10 --num-filter-maps 50 --dropout 0.2 --patience 10 --criterion prec_at_8 --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
#跑40个epoch，卷积核数量翻4倍，性能有明显改进
python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 conv_attn 40 --filter-size 10 --num-filter-maps 200 --dropout 0.2 --patience 10 --criterion prec_at_8 --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu
