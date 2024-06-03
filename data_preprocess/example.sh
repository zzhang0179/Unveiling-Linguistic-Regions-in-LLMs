
# 循环遍历语言列表并执行相应的Python脚本
mkdir -p path_to_save
python preprocess-llama.py \
    --mode "write" \
    --file_path "example.jsonl" \
    --save_prefix "train" \
    --save_path "path_to_save/" \
    --language "chinese" \
    --do_keep_newlines \
    --seq_length 512 \
    --tokenizer_path 'LLaMA-2-Tokenizer' \
    --num_workers 16 \

python preprocess-llama.py \
    --mode="read" \
    --read_path_prefix="./path_to_save/train" \
    --tokenizer_path 'LLaMA-2-Tokenizer'