export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/


DATASET=sst2
N_TRAIN_SHOT=4
N_DEMO_SHOT=1
for SEED in 1 2 3 4 5; do
python3 fads-icl.py \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_train_shot ${N_TRAIN_SHOT} \
        --n_demo_shot ${N_DEMO_SHOT} \
        --seed ${SEED} \
        --output_dir ./output/fads-icl/${LLM} \
        --feature_choose all
done

# for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do


# N_DEMO_SHOT=1
# for N_TRAIN_SHOT in 4 8 16 32 64 128; do
#     for SEED in 1 2 3 4 5; do
#     python3 fads-icl.py \
#         --llm_dir ${LLM_DIR} \
#         --dataset ${DATASET} \
#         --data_dir ${DATA_DIR} \
#         --n_train_shot ${N_TRAIN_SHOT} \
#         --n_demo_shot ${N_DEMO_SHOT} \
#         --seed ${SEED} \
#         --output_dir ./output/fads-icl/${LLM} \
#         --feature_choose all
#     done
# done

# done
