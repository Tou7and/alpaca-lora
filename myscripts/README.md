# Some Notes about Custom Fine-tuning
2023.10.12, to make training success:
- Use older peft version (peft==0.3.0.dev0) instead of current newest (peft==0.5.0)
- V100 GPU does not support 8-bit training, so turn off all the int8 options in the training script, if training with V100 or older GPU.
- Use [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf) as base model instead of [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

PS:
```
The adapter model(bin file) should be in "mb" level size.
But there is a problem in peft 0.5.0, which lead to incorrect adapter model file (which is "kb" level size)
According to reference, the easy way to fix this is going back to the old PEFT version:
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08

List my files under the trained model folder for reference:
4.0K    models/lora-alpaca-tw-v3/checkpoint-6200/adapter_config.json
65M     models/lora-alpaca-tw-v3/checkpoint-6200/adapter_model.bin
129M    models/lora-alpaca-tw-v3/checkpoint-6200/optimizer.pt
16K     models/lora-alpaca-tw-v3/checkpoint-6200/rng_state.pth
4.0K    models/lora-alpaca-tw-v3/checkpoint-6200/scheduler.pt
80K     models/lora-alpaca-tw-v3/checkpoint-6200/trainer_state.json
4.0K    models/lora-alpaca-tw-v3/checkpoint-6200/training_args.bin

If one use V100, or any other older GPU, turn off the all the "int8" configurations in finetune.py (see finetune_v100.py for detail.) 

yahma/llama-7b-hf resolve the EOS token issues of decapoda-research/llama-7b-hf.
(Is it beacuse you're using yahma/llama-7b-hf instead of decapoda-research/llama-7b-hf ?
Yes, If the decapoda-research/llama-7b-hf is still using ros=bos=0, that might be the problem. 
Because the model doesn’t know how to stop, unless we use the llama data to finetune the whole model again.)
```

Reference:
- https://github.com/tloen/alpaca-lora/issues/367
- https://github.com/tloen/alpaca-lora/issues/524
- https://github.com/tloen/alpaca-lora/issues/526
- https://github.com/tloen/alpaca-lora/issues/334

## Chinese Data
- [traditional-chinese-alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca/tree/main) provide its data and recipes for training TW alpaca.

