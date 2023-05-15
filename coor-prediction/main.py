import argparse
import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)
from predict_coor import (
    gen_coor_lm,
    gen_coor_alpaca
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict distances'
    )
    parser.add_argument(
        '--model',
        dest='model_name',
        choices=[
            'opt',
            'llama',
            'alpaca'
        ],
        required=True,
    )
    parser.add_argument(
        '--model_size',
        dest='model_size',
        choices=[
            '0',
            '1',
        ],
        default='0',
    )
    parser.add_argument(
        '--local',
        dest='local',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--p_type',
        dest='p_type',
        choices=[
            '0',
            '1',
        ],
        required=True,
    )
    parser.add_argument(
        '--p_length',
        dest='p_length',
        choices=[
            'zero-shot',
            '3-shot',
        ],
        required=True,
    )
    args = parser.parse_args()
    print(args)

    cities = pd.read_pickle('worldcities100k.pkl')

    if args.local:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                        "MPS not available because the current PyTorch install"
                        " was not built with MPS enabled."
                    )
            else:
                print(
                        "MPS not available because the current MacOS version "
                        "is not 12.3+ and/or you do not have an MPS-enabled "
                        "device on this machine."
                    )

        else:
            # device = torch.device("mps")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if args.model_name == 'opt':
        if args.local:
            model_name = "facebook/opt-350m"
        else:
            if args.model_size == '0':
                model_name = "facebook/opt-6.7b"
            elif args.model_size == '1':
                model_name = "facebook/opt-13b"
        model = OPTForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        result = gen_coor_lm(
            model,
            tokenizer,
            device,
            model_name,
            cities,
            args.p_type,
            args.p_length
        )
        file_path = (
                f'outputs/gen_coor_{model_name.split("/")[1]}-{args.p_length}-'
                f'{args.p_type}.json'
        )
        json.dump(
            result,
            open(
                file_path,
                'w+'
            )
        )
    elif args.model_name == 'llama':
        if args.local:
            raise Exception('llama not available locally')
        else:
            if args.model_size == '0':
                model_name = "facebook/llama-7b"
                model_loc = '/scratch/pbhanda2/projects/llama/hf_llama/7B'
            elif args.model_size == '1':
                model_name = "facebook/llama-13b"
                model_loc = '/scratch/pbhanda2/projects/llama/hf_llama/13B'
            model = LlamaForCausalLM.from_pretrained(
                model_loc
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                model_loc
            )
        model.to(device)
        result = gen_coor_lm(
            model,
            tokenizer,
            device,
            model_name,
            cities,
            args.p_type,
            args.p_length
        )
        file_path = (
                f'outputs/gen_coor_{model_name.split("/")[1]}-{args.p_length}-'
                f'{args.p_type}.json'
        )
        json.dump(
            result,
            open(
                file_path,
                'w+'
            )
        )
    elif args.model_name == 'alpaca':
        if args.local:
            raise Exception('alpaca not available locally')
        else:
            model_name = 'stanford/alpaca-7b'
            model = LlamaForCausalLM.from_pretrained(
                '/scratch/pbhanda2/projects/llama/alpaca_weights_7B'
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                '/scratch/pbhanda2/projects/llama/alpaca_weights_7B'
            )
        model.to(device)
        result = gen_coor_alpaca(
            model,
            tokenizer,
            device,
            model_name,
            cities,
            args.p_type
        )
        file_path = (
                f'outputs/gen_coor_{model_name.split("/")[1]}-{args.p_length}-'
                f'{args.p_type}.json'
        )
        json.dump(
            result,
            open(
                file_path,
                'w+'
            )
        )
