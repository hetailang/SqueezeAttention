import argparse
import sys
import os
import json, tqdm
import torch
import copy


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
from modify_opt_my import convert_kvcache_opt_test, OPTDecoder_my

def convert_nochange(model, config):
    print("We do nothing to this model")
    return model
ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
    "test": convert_kvcache_opt_test,
    "no_change": convert_nochange,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default='input.jsonl')
    parser.add_argument('--output-path', type=str, default='output_step2.jsonl')
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument("--cache-dir", type=str, default='../../checkpoint/')
    parser.add_argument("--output-file", type=str, default='result')

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--start_layer", type=int, default=23)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    output_file = args.output_file
    model_path = '/home/user/.cache/huggingface/hub/models--facebook--opt-350m'

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio
    config.start = args.start_layer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    checkpoint = copy.deepcopy(model.state_dict())
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
    model.load_state_dict(checkpoint)
    print(model)

    st = config.start
    with open(str(output_file), 'a+') as fw:
        tt = 'starting from layer '+ str(st) + 'the following layeres('+ str(config.num_hidden_layers - st) +  ') are the same as layer ' + str(st - 1) 
        print(tt, file=fw)
    #    print(model, file=fw)
    

    if args.enable_small_cache:
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        model.load_state_dict(checkpoint)
        print(model)

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            logits = model(input_ids).logits.log_softmax(dim=-1)

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
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

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
