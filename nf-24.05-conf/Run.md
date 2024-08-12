```

```

```
# ref: https://github.com/google/sentencepiece
cd /opt
git clone https://github.com/google/sentencepiece.git


mkdir -p /workspace/data/mm/llama2/7b/llama-2-7b-chat-hf/neva/tokenizers
cd /opt/sentencepiece/src/; protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
--input_file /workspace/data/mm/llama2/7b/llama-2-7b-chat-hf/tokenizer.model \
--output_file /workspace/data/mm/llama2/7b/llama-2-7b-chat-hf/neva/tokenizers/tokenizer_neva.model \
--is_userdefined \
--tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
         "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"
```

TODO LIST:



