import argparse
import json
import torch
import pandas as pd
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
from predict_coor import (
    gen_coor_lm_usa,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict distances'
    )
    parser.add_argument(
        '--local',
        dest='local',
        type=bool,
        default=False
    )
    args = parser.parse_args()
    print(args)

    cities = pd.read_pickle('../coor-prediction-from-distance/cities.pkl')

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
    result = gen_coor_lm_usa(
        model,
        tokenizer,
        device,
        model_name,
        cities,
    )
    file_path = (
            f'outputs/gen_coor_usa_{model_name.split("/")[1]}.json'
    )
    json.dump(
        result,
        open(
            file_path,
            'w+'
        )
    )
