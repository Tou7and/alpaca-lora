# Some Notes about Custom Fine-tuning
2023.10.12, to make training success:
- Use older peft version (peft==0.3.0.dev0) instead of current newest (peft==0.5.0)
- Use [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf) instead of [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)
- The adapter model(bin file) is in "kb" level size (it should be in mb level, according to the reference).
- V100 GPU does not support 8-bit training

## Reference:
- https://github.com/tloen/alpaca-lora/issues/367
```
Tested your script and seems to fix the issue, thank you! Is it beacuse you're using yahma/llama-7b-hf instead of decapoda-research/llama-7b-hf ?
Yes, If the decapoda-research/llama-7b-hf is still using ros=bos=0, that might be the problem. Because the model doesn’t know how to stop, unless we use the llama data to finetune the whole model again.
```
- https://github.com/tloen/alpaca-lora/issues/524
- https://github.com/tloen/alpaca-lora/issues/526
- https://github.com/tloen/alpaca-lora/issues/334
```
I just tried with 10 instance, and it generated 67M adapter_model.bin.
Or you can use the easy way, back to the old PEFT version:
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```

## Chinese Data
- [traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca/tree/main) provide its data and recipes for training TW alpaca.


