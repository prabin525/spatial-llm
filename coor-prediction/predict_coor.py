import torch
import json
import copy
from typing import (
    List,
    Literal,
    # Optional,
    # Tuple,
    TypedDict,
    # Union
)


Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]"
        to your inputs. Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} "
                    f"{E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def gen_coor_lm(
        model,
        tokenizer,
        device,
        model_name,
        cities,
        p_type=0,
        p_length='zero-shot'
):

    results = []
    template = [
        open('templates/LM0-zero-shot.txt').read(),
        open('templates/LM1-zero-shot.txt').read(),
        open('templates/LM0-3-shot.txt').read(),
        open('templates/LM1-3-shot.txt').read(),
    ]

    for i, each in cities.iterrows():
        res = each.to_dict()
        if p_length == 'zero-shot':
            if p_type == '0':
                prompt = template[0].format(
                        city=each.AccentCity,
                    )

            elif p_type == '1':
                prompt = template[1].format(
                        city=each.AccentCity,
                    )
        elif p_length == '3-shot':
            remaining = cities.loc[cities.index != i]
            sampled_city = remaining.sample(3)
            if p_type == '0':
                prompt = template[2].format(
                        city_a=sampled_city.iloc[0].AccentCity,
                        lat_a=str(sampled_city.iloc[0].Latitude),
                        lng_a=str(sampled_city.iloc[0].Longitude),
                        city_b=sampled_city.iloc[1].AccentCity,
                        lat_b=str(sampled_city.iloc[1].Latitude),
                        lng_b=str(sampled_city.iloc[1].Longitude),
                        city_c=sampled_city.iloc[2].AccentCity,
                        lat_c=str(sampled_city.iloc[2].Latitude),
                        lng_c=str(sampled_city.iloc[2].Longitude),
                        city=each.AccentCity,
                    )

            elif p_type == '1':
                prompt = template[3].format(
                        city_a=sampled_city.iloc[0].AccentCity,
                        lat_a=str(sampled_city.iloc[0].Latitude),
                        lng_a=str(sampled_city.iloc[0].Longitude),
                        city_b=sampled_city.iloc[1].AccentCity,
                        lat_b=str(sampled_city.iloc[1].Latitude),
                        lng_b=str(sampled_city.iloc[1].Longitude),
                        city_c=sampled_city.iloc[2].AccentCity,
                        lat_c=str(sampled_city.iloc[2].Latitude),
                        lng_c=str(sampled_city.iloc[2].Longitude),
                        city=each.AccentCity,
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
            max_new_tokens=60,
            # eos_token_id=tokenizer.encode('\n\n')
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        res.update({
            'output': outputs[0].replace(prompt, ''),
            'full_output': outputs[0],
            'model': model_name.split('/')[1].split('-')[0],
            'model_size': model_name.split('/')[1].split('-')[1],
            'p_type': p_type,
            'p_length': p_length
        })
        results.append(res)
    return results


def gen_coor_alpaca(
    model,
    tokenizer,
    device,
    model_name,
    cities,
    p_type=0,
):
    results = []
    template = [
        open('templates/Alpaca0.txt').read(),
        open('templates/Alpaca1.txt').read(),
    ]

    for i, each in cities.iterrows():
        res = each.to_dict()
        if p_type == '0':
            prompt = template[0].format(
                    city=each.AccentCity,
                )

        elif p_type == '1':
            prompt = template[1].format(
                    city=each.AccentCity,
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
            max_new_tokens=60,
            # eos_token_id=tokenizer.encode('\n\n')
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        res.update({
            'output': outputs[0].replace(prompt, ''),
            'full_output': outputs[0],
            'model': model_name.split('/')[1].split('-')[0],
            'model_size': model_name.split('/')[1].split('-')[1],
            'p_type': p_type,
        })
        results.append(res)
    return results


def gen_coor_lm_usa(
        model,
        tokenizer,
        device,
        model_name,
        cities,
):

    results = []
    template = [
        open('templates/LM1-zero-shot.txt').read(),
    ]
    cities = list(set(cities.a_name.to_list()))

    for each in cities:
        res = {
            'name': each
        }
        prompt = template[0].format(
                city=each,
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
            max_new_tokens=60,
            # eos_token_id=tokenizer.encode('\n\n')
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        res.update({
            'output': outputs[0].replace(prompt, ''),
            'full_output': outputs[0],
            'model': model_name.split('/')[1].split('-')[0],
            'model_size': model_name.split('/')[1].split('-')[1],
        })
        results.append(res)
    return results


def gen_coor_llama2_chat(
    model,
    tokenizer,
    device,
    model_name,
    cities,
    p_type=0,
):
    results = []
    template = [
        json.load(open('templates/LLAMA2-CHAT-0.json')),
        json.load(open('templates/LLAMA2-CHAT-1.json')),
    ]

    for i, each in cities.iterrows():
        res = each.to_dict()
        if p_type == '0':
            prompt = copy.deepcopy(template[0])
            prompt[0][1]['content'] = prompt[0][1]['content'].format(
                                        city=each.AccentCity
                                    )

        elif p_type == '1':
            prompt = copy.deepcopy(template[1])
            prompt[0][1]['content'] = prompt[0][1]['content'].format(
                                        city=each.AccentCity
                                    )

        chat = format_tokens(prompt, tokenizer)
        tokenized_prompt = torch.tensor(chat[0]).long()
        tokenized_prompt = tokenized_prompt.unsqueeze(0)
        tokenized_prompt = tokenized_prompt.to(device)
        gen = model.generate(
            tokenized_prompt,
            do_sample=False,
            num_beams=5,
            max_new_tokens=100,
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        res.update({
            'output': outputs[0].split('[/INST]')[1].strip(),
            'full_output': outputs[0],
            'model': model_name.split('/')[1].split('-')[0],
            'model_size': model_name.split('/')[1].split('-')[1],
            'p_type': p_type,
        })
        results.append(res)
    return results
