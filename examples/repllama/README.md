# RepLLaMA retrieval example

> code tested on transformers==4.33.0 peft==0.4.0 A6000 GPU

In this example, we show how to use RepLLaMA to run zeroshot evaluation on BEIR, and how to train RepLLaMA from scratch.


## Inference with RepLLaMA
For the following steps, we use SciFact in BEIR as example.

### Increase swap for CPU-running encoding using full model
The minimum requirement is 32GB memory + 16GB swap.
To create an additional 8GB swap file (will be removed after system reboot):
```
sudo fallocate -l 8G /swapfile
sudo dd if=/dev/zero of=/swapfile bs=1024 count=8388608
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
sudo free -h
```

Encoding a scifact passage requires 130s.

### Encode corpus
```
mkdir beir_embedding_scifact
for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=0 python encode.py \
  --output_dir=temp \
  --model_name_or_path castorini/repllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --fp16 \
  --per_device_eval_batch_size 16 \
  --p_max_len 512 \
  --dataset_name Tevatron/beir-corpus:scifact \
  --encoded_save_path beir_embedding_scifact/corpus_scifact.${s}.pkl \
  --encode_num_shard 4 \
  --encode_shard_index ${s}
done
```
> We shard the encoding, so that it can be parallelized on multiple GPUs, when its available.

### Encode queries
```
CUDA_VISIBLE_DEVICES=0 python encode.py \
  --output_dir=temp \
  --model_name_or_path castorini/repllama-v1-7b-lora-passage \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --fp16 \
  --per_device_eval_batch_size 16 \
  --q_max_len 512 \
  --dataset_name Tevatron/beir:scifact/test \
  --encoded_save_path beir_embedding_scifact/queries_scifact.pkl \
  --encode_is_qry
```

### Search

python -m tevatron.retriever.driver.search \
    --query_reps beir_embedding_scifact/queries_scifact.pkl \
    --passage_reps 'beir_embedding_scifact/corpus_scifact.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to beir_embedding_scifact/rank.scifact.txt

### Convert to TREC format
```
python -m tevatron.utils.format.convert_result_to_trec --input beir_embedding_scifact/rank.scifact.txt \
                                                       --output beir_embedding_scifact/rank.scifact.trec \
                                                       --remove_query
```

### Evaluate
```
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test beir_embedding_scifact/rank.scifact.trec
```

## Train RepLLaMA from scratch

This example demonstrate train repllama from the base LLaMA2 model using MS MARCO passage training data
```
deepspeed --include localhost:0,1,2,3 train.py \
  --deepspeed ds_config.json \
  --output_dir model_repllama \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --save_steps 200 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --train_n_passages 16 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --dataset_proc_num 32 \
  --negatives_x_device \
  --warmup_steps 100
```


### Citation
```
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
```