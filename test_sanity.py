"""
Sanity check testing.
- memory usage: 31821MiB / 32510MiB (V100*1)

Decoding config (https://huggingface.co/docs/transformers/main_classes/text_generation)
- greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
- contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
- multinomial sampling by calling sample() if num_beams=1 and do_sample=True
- beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
- beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
- diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
- constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
- assisted decoding by calling assisted_decoding(), if assistant_model is passed to .generate()

2023.10.09.
"""
import os
import sys
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import os.path as osp
from typing import Union
from tqdm import tqdm

output_path = "data/testing/sanitycheck-tw-v3.json"
load_8bit = False
base_model = "/media/volume1/aicasr/yahma/llama-7b-hf"
lora_weights = "/home/t36668/projects/tou7and/alpaca-lora/models/lora-alpaca-tw-v3/checkpoint-6200"
# lora_weights = "/home/t36668/projects/tou7and/alpaca-lora/models/lora-alpaca-cmuh-v1c/checkpoint-19800"
device = "cuda"
prompt_template = "alpaca" # prompt_template: str = "",  # The prompt template to use, will default to alpaca.

p1 = "Based on the present illness and hospital course, predict the discharge diagnosis."
x1 = """
This 67 year old lady has well relative health before.. Denied other major disease. Nither TOCC
history. According to the family, the patient suffered from sore throat for 3 days. Productive
cough, dysphagia, body weight loss were also noted. There were no headache, dizziness, chest
tightness, palpitation, abdomen pain, diarrhea, constipation, nor urinary problems. He was brought
to our emergency department for help. Esophageal cancer was suspected. The patient was admitted for
further evaluation and treatment.
"""
test_data = [
    {"instruction": p1, "input": x1},
    {"instruction": "Find corresponding ICD codes", "input": "1.Malignant neoplasm of esophagus, unspecified 2.Malignant neoplasm of hypopharynx, unspecified"},
    {"instruction": "Write a regular expression that can find english words from a sentence", "input": None},
    {"instruction": "請你根據以下的主題寫一首詩", "input": "白雲飄飄"},
    {"instruction": "有機物和無機物的差異是什麼", "input": None},
    {"instruction": "告訴我三種賺錢的方式", "input": None},
    {"instruction": "請解下列一元方程式。", "input": "x+3=7, x=?"},
]

test_data = [
    {"instruction": "台灣的領導人是誰", "input": ""},
    {"instruction": "台灣的國慶日是幾月幾號", "input": ""},
    {"instruction": "中國的領導人是誰", "input": ""},
    {"instruction": "中國的國慶日是幾月幾號", "input": ""},
    {"instruction": "Who is the president of Taiwan?", "input": ""},
    {"instruction": "Who is the president of China?", "input": ""},
]

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)


model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.half()
model.eval()

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def evaluate(instruction, input=None, max_new_tokens=128):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # print(inputs['input_ids'][0].shape)
    input_ids = inputs["input_ids"].to(device)
    
    generation_config = GenerationConfig(do_sample=False, num_beams=1) # Greedy search
    # generation_config = GenerationConfig(do_sample=True, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, **kwargs)

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    # remove:  "max_new_tokens": max_new_tokens,

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    outputs = tokenizer(output, return_tensors="pt")
    # print(outputs['input_ids'][0].shape)
    return prompter.get_response(output)

for question in tqdm(test_data):
    output = evaluate(question['instruction'], question['input'], max_new_tokens=500)
    question['output'] = output

with open(output_path, "w", encoding="utf-8") as writer:
    json.dump(test_data, writer, indent=4, ensure_ascii=False)
