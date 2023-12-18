import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import copy
from utils_hh.modify_mistral import convert_kvpruner_mistral, MistralModel_my
from utils_hh.modify_llama import convert_kvpruner_llama, LlamaModel_my, LlamaAttention_my
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def convert_nochange(model, config):
    print('We do nothing to this model')
    return model

ENABLE_KV_Pruner_FUNCTIONS = {
        "no_change": convert_nochange,
        "Mistral": convert_kvpruner_mistral,
        "llama": convert_kvpruner_llama,
        }
TAGET_MODULE = {
        'Mistral': MistralModel_my,
        'llama': LlamaAttention_my,
        "no_change": MistralModel_my, # for debug
        }
DecoderLayer = {
        "llama": "LlamaDecoderLayer_my",
        "Mistral": "MistralDecoderLayer_my",
        "no_change": "LlamaDecoderLayer",  # for debug
        }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Mistral", choices=["Mistral", "llama2-13B", "llama2-70B"])
    parser.add_argument('--pred', type=str, choices=['Long', 'helm'])
    parser.add_argument('--hiddlayer', action='store_true', help='whether to record the output about hidden layers')
    parser.add_argument('--model_arch', type=str, default='Mistral', choices=["llama", "Mistral", "no_change"], help='default:%(default)s')
    parser.add_argument('--device', type=str, default='cuda:0', help='default=%(default)s')
    parser.add_argument('--sample_num', type=int, default=200, help='sample number in Helm, default=%(default)s')
    parser.add_argument('--ini_size', type=float, help='initial window size')
    parser.add_argument('--percent', type=float, default=10, help='the percent of the minimum size, default=%(default)s')
    parser.add_argument('--exp', type=int, default=10, help='which function to choose, default=%(default)s')
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def record_hidd_data(model, model_arch):
    for name, m in model.named_modules():
        if isinstance(m, TAGET_MODULE[model_arch]):
            m.record_hidd_data()

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, args):
    device = args.device
    model_name = args.model
    model_arch = args.model_arch
    hiddlayer = args.hiddlayer
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})

#        if hiddlayer == True:
#            record_hidd_data(model, model_arch)
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, args):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.hiddlayer = args.hiddlayer
    if args.model == 'Mistral':
        config._flash_attn_2_enabled = True
    config.ini_size = args.ini_size
    config.percent = args.percent
    config.exp = args.exp
    device = args.device
    model_arch = args.model_arch
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    if args.model == 'llama2-13B':
        device_map = {'model.embed_tokens': 2, 'model.layers.0': 2, 'model.layers.1': 2, 'model.layers.2': 2, 'model.layers.3': 2, 'model.layers.4': 2, 'model.layers.5': 2, 'model.layers.6': 2, 'model.layers.7': 2, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2 , 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 3, 'model.layers.22': 3, 'model.layers.23': 3, 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 3, 'model.layers.35': 3, 'model.layers.36': 3, 'model.layers.37': 3, 'model.layers.38': 3, 'model.layers.39': 3, 'model.norm': 3, 'lm_head': 3}
    else:
#        device_map = 'auto'
        device_map = infer_auto_device_map(model, max_memory={int(device[-1]): "48GiB"})
    model = ENABLE_KV_Pruner_FUNCTIONS[model_arch](model, config)
    if model_arch == 'no_change':
        model_arch = 'llama' if 'llama' in args.model else 'Mistral'
    print('After change:')
#    print(model)
    model = load_checkpoint_and_dispatch(
    model, path, device_map=device_map, no_split_module_classes=[DecoderLayer[model_arch]], dtype=torch.bfloat16)
    print(model.hf_device_map)
    return model, tokenizer

def get_dataset(args):
    if args.pred == 'Long':
        print('running with model', args.model, 'in Long using', args.device)
        datasets = ['samsum']
    elif args.pred == 'helm':
        print('running with model ', args.model, 'in helm, using', args.device)
        datasets = ['xsum']
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    return datasets

def get_data_and_outpath(args, model_name, dataset):
    if args.pred == 'helm':
        input_path = '/home/user/wangzihao/H2O-main/h2o_hfm/data/xsum.jsonl'
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip() != '':
                    data.append(json.loads(line))
    else:
        data = load_dataset('/home/user/.cache/huggingface/datasets/longbench/LongBench.py', dataset, split='test')

    if not os.path.exists(f"pred_{args.pred}/{model_name}/func_{args.exp}/{args.ini_size}/{int(args.percent * 100)}"):
        os.makedirs(f"pred_{args.pred}/{model_name}/func_{args.exp}/{args.ini_size}/{int(args.percent * 100)}")

    out_path = f"pred_{args.pred}/{model_name}/func_{args.exp}/{args.ini_size}"
    out_path += f"/{int(args.percent * 100)}"
    out_path += f"/{dataset}.jsonl" 
    return data, out_path

def helm(model, tokenizer, requests, args):
    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]
    
    args.k = 0
    model_arch = args.model_arch

    results = []
    with torch.no_grad():
        for request in tqdm(requests):
            request = request['request']
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                #max_new_tokens=64,
                temperature=temperature,
                top_k=args.k,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            for name, m in model.named_modules():
                if isinstance(m, TAGET_MODULE[args.model_arch]):
                    m._reset_masks()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            result['result'] = {
                "choices": [
                    {
                        "text": generate_text,
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)
    return results

def check_args(args):
    assert args.model == 'Mistral' and args.pred == 'Long' or \
           'llama2' in args.model and args.pred == 'helm', "Please use llama2 on helm and Mistral on LongBench"



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    check_args(args)
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model

    # define your model

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], args)
    max_length = model2maxlen[model_name]

    datasets = get_dataset(args)

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    for dataset in datasets:
        data, out_path = get_data_and_outpath(args, model_name, dataset)
        out_file = out_path[:out_path.rfind('/')]
        assert os.path.exists(out_file), "%s doesn't exits" % out_file
        
        if args.pred == 'helm':
            preds = helm(model, tokenizer, data, args)
        else:
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, args)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
