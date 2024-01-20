import argparse
import json
import torch
import pandas as pd
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)


def gen_dis(
        model,
        tokenizer,
        device,
        cities,
        p_length
):

    results = []
    template = [
        open('templates/dis-zero-shot.txt').read(),
        open('templates/dis-3-shot.txt').read(),
    ]

    for i, each in cities.iterrows():
        outputs = []
        res = each.to_dict()
        if p_length == 'zero-shot':
            prompt = template[0].format(
                    city_a=each["a_name"],
                    city_b=each["b_name"],
                )
        elif p_length == '3-shot':
            sampled_cities = cities.loc[
                (cities.a_name != each['a_name']) &
                (cities.a_name != each['b_name']) &
                (cities.b_name != each['a_name']) &
                (cities.b_name != each['b_name'])
            ].sample(3).reset_index()
            prompt = template[1].format(
                    city_a=sampled_cities.iloc[0]["a_name"],
                    city_b=sampled_cities.iloc[0]["b_name"],
                    dis_a=sampled_cities.iloc[0]["distance"],
                    city_c=sampled_cities.iloc[1]["a_name"],
                    city_d=sampled_cities.iloc[1]["b_name"],
                    dis_b=sampled_cities.iloc[1]["distance"],
                    city_e=sampled_cities.iloc[2]["a_name"],
                    city_f=sampled_cities.iloc[2]["b_name"],
                    dis_c=sampled_cities.iloc[2]["distance"],
                    city_g=each["a_name"],
                    city_h=each["b_name"],
                )

        tokenized_prompt = tokenizer.encode(
            prompt,
            return_tensors='pt'
        )
        tokenized_prompt = tokenized_prompt.to(device)
        gen = model.generate(
            tokenized_prompt,
            do_sample=False,
            num_beams=5,
            max_new_tokens=5,
        )
        o = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        outputs.extend(o)
        res.update({
            'output': outputs,
            'prompt': prompt,
        })
        results.append(res)
    return results


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
        '--p_length',
        dest='p_length',
        choices=[
            'zero-shot',
            '3-shot',
        ],
        required=True,
    )
    parser.add_argument(
        '--use_llama2',
        dest='use_llama2',
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    print(args)

    cities = pd.read_pickle('../spatial-awareness/count_df.pkl')

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
        if args.use_llama2:
            model_name = "meta-llama/Llama-2-70b-hf"
            access_token = open('access_token').read()
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=access_token,
                load_in_8bit=True,
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                model_name
            )
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
    result = gen_dis(
        model,
        tokenizer,
        device,
        cities,
        args.p_length
    )
    if args.use_llama2:
        file_path = f'outputs/gen_dis_llama2-{args.p_length}.json'
    else:
        file_path = f'outputs/gen_dis-{args.p_length}.json'
    json.dump(
        result,
        open(
            file_path,
            'w+'
        )
    )
