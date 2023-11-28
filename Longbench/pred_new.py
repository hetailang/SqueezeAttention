import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import copy
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from utils_hh.modify_mistral import convert_kvpruner_mistral, get_sliding_windows, MistralModel_my
from utils_hh.modify_llama import convert_kvpruner_llama, LlamaModel_my
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map

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
        'llama': LlamaModel_my,
        }
DecoderLayer = {
        "llama": "LlamaDecoderLayer_my",
        "Mistral": "MistralDecoderLayer_my"
        }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Mistral", choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k",\
            "vicuna-v1.5-7b-16k", "gpt2", "opt-1.3b", "Mistral", "llama7B", "llama2-13B"])
    parser.add_argument('--pred', type=str, choices=['e', 'en', 'test', 'exp', 'imp', 'helm'])
    parser.add_argument('--hiddlayer', action='store_true', help='whether to record the output about hidden layers')
    parser.add_argument('--model_arch', type=str, default='Mistral', choices=["llama", "Mistral"], help='default:%(default)s')
    parser.add_argument('--mode', type=str, default='default', help='the mode about sliding_windows, default=%(default)s')
    parser.add_argument('--spec_size', type=int, default=0, help='When it is necessary to specify a window size, use this argu')
    parser.add_argument('--min_size', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0', help='default=%(default)s')
    parser.add_argument('--sample_num', type=int, default=200, help='sample number in Helm, default=%(default)s')
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

def prepare_before_generate(model, prompt_len, max_tokens, model_arch, mode, spec_size, min_size):
    if model_arch == 'no_change':
        return
    for name, m in model.named_modules():
        if isinstance(m, TAGET_MODULE[model_arch]):
            m.reset_siding_windows(prompt_len, max_tokens, mode, spec_size, min_size)

def record_hidd_data(model, model_arch):
    for name, m in model.named_modules():
        if isinstance(m, TAGET_MODULE[model_arch]):
            m.record_hidd_data()

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, args):
    device = args.device
    model_name = args.model
    model_arch = args.model_arch
    mode = args.mode
    spec_size = args.spec_size
    min_size = args.min_size
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

        prepare_before_generate(model, context_length, max_gen, model_arch, mode, spec_size, min_size)

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

        if hiddlayer == True:
            record_hidd_data(model, model_arch)
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, model_arch, hiddlayer):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.hiddlayer = hiddlayer
    config._flash_attn_2_enabled = True
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    if model_name == 'llama2-13B':
        device_map = {'model.embed_tokens': 2, 'model.layers.0': 2, 'model.layers.1': 2, 'model.layers.2': 2, 'model.layers.3': 2, 'model.layers.4': 2, 'model.layers.5': 2, 'model.layers.6': 2, 'model.layers.7': 2, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2 , 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 3, 'model.layers.22': 3, 'model.layers.23': 3, 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 3, 'model.layers.35': 3, 'model.layers.36': 3, 'model.layers.37': 3, 'model.layers.38': 3, 'model.layers.39': 3, 'model.norm': 3, 'lm_head': 3}
    else:
        device_map = 'sequential'
    model = ENABLE_KV_Pruner_FUNCTIONS[model_arch](model, config)
    model = load_checkpoint_and_dispatch(
    model, path, device_map=device_map, no_split_module_classes=[DecoderLayer[model_arch]], dtype=torch.bfloat16)
    print('After change:')
#    print(model)
#    print(model.hf_device_map)
    return model, tokenizer

def get_dataset(args):
    if args.pred == 'e':
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    elif args.pred == 'test':
        print('running with model ', args.mode, 'in test, using', args.device)
        datasets = ["samsum"]
    elif args.pred == 'en':
        print('running with model ', args.mode, 'in en, using', args.device)
        datasets = ["2wikimqa", "gov_report", "hotpotqa", "lcc", "multi_news", "multifieldqa_en", "musique",\
                "narrativeqa", "passage_count", "passage_retrieval_en", "qasper", "qmsum", "repobench-p", "samsum",\
                "trec", "triviaqa"]
    elif args.pred == 'exp':
        print('running with model ', args.mode, 'in exp, using', args.device)
        datasets = ['triviaqa', 'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p']
    elif args.pred == 'imp':
        print('running with model ', args.mode, 'in imp, using', args.device)
        datasets = ['samsum']
    elif args.pred == 'helm':
        print('running with model ', args.mode, 'in helm, using', args.device)
        datasets = ['xsum']
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    if args.min_size != 0:
        print('using min_size to improve precision')
    return datasets

def get_data_and_outpath(args, model_name, mode, dataset):
    if args.pred == 'e':
        data = load_dataset('/home/user/.cache/huggingface/datasets/longbench/LongBench.py', f"{dataset}_e", split='test')
    elif args.pred == 'helm':
        input_path = '/home/user/wangzihao/H2O-main/h2o_hfm/data/xsum.jsonl'
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip() != '':
                    data.append(json.loads(line))
    else:
        data = load_dataset('/home/user/.cache/huggingface/datasets/longbench/LongBench.py', dataset, split='test')

    if not os.path.exists(f"pred_{args.pred}/{model_name}/{mode}"):
        os.makedirs(f"pred_{args.pred}/{model_name}/{mode}")
    if args.spec_size != 0 and not os.path.exists(f"pred_{args.pred}/{model_name}/{mode}/{args.spec_size}"):
        os.makedirs(f"pred_{args.pred}/{model_name}/{mode}/{args.spec_size}")

    out_path = f"pred_{args.pred}/{model_name}/{mode}"
    if args.spec_size != 0:
        out_path += f"/{args.spec_size}"
    out_path += f"/{dataset}.jsonl"
    
    return data, out_path

def check_mode(mode, args):
    if mode == 'spec':
        assert args.spec_size != 0, "when useing sepc mode, need a argument sepc-size"

def helm(model, tokenizer, requests, args):
    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]
    
    args.k = 0
    model_arch = args.model_arch
    mode = args.mode
    spec_size= args.spec_size
    min_size = args.min_size

    results = []
    with torch.no_grad():
        for request in tqdm(requests):
            request = request['request']
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            context_length = input_ids.shape[-1]
            prepare_before_generate(model, context_length, request['max_tokens'], model_arch, mode, spec_size, min_size)

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



if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device
    model_name = args.model
    mode = args.mode
    min_size = args.min_size
    model_arch = args.model_arch
    spec_size = args.spec_size
    hiddlayer = args.hiddlayer
    check_mode(mode, args)
    # define your model
#    dataset = 'samsum'
#    data, out_path = get_data_and_outpath(args, model_name, mode, dataset)
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args.model_arch, hiddlayer)
    max_length = model2maxlen[model_name]

    datasets = get_dataset(args)

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset if not os.path.exists("pred"):
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:

        data, out_path = get_data_and_outpath(args, model_name, mode, dataset)
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
