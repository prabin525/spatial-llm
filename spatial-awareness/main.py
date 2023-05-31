import argparse
import json
import torch
import pandas as pd
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
from gen_sen import (
    gen_sen,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate spatially aware sentences'
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
            'near',
            'far',
            'close',
            'and'
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
    parser.add_argument(
        '--state',
        dest='state',
        choices=[
            'true',
            'false',
        ],
        required=True,
    )
    args = parser.parse_args()
    print(args)

    if args.state == 'true':
        state = True
    else:
        state = False

    cities = pd.read_pickle('cities.pkl')

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
            device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if args.local:
        raise Exception('llama not available locally')
    else:
        model_name = "facebook/llama-13b"
        model_loc = '/scratch/pbhanda2/projects/llama/hf_llama/13B'
        model = LlamaForCausalLM.from_pretrained(
            model_loc
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            model_loc
        )
    model.to(device)
    result = gen_sen(
        model,
        tokenizer,
        device,
        cities,
        args.p_type,
        args.p_length,
        state
    )
    state_rec = 'no-state'
    if state is True:
        state_rec = 'state'
    file_path = (
            f'outputs/gen_sen-{args.p_length}-{args.p_type}-'
            f'{state_rec}.json'
    )
    json.dump(
        result,
        open(
            file_path,
            'w+'
        )
    )
