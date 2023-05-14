
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
                        city_a=sampled_city.iloc[1].AccentCity,
                        lat_a=str(sampled_city.iloc[1].Latitude),
                        lng_a=str(sampled_city.iloc[1].Longitude),
                        city_a=sampled_city.iloc[2].AccentCity,
                        lat_a=str(sampled_city.iloc[2].Latitude),
                        lng_a=str(sampled_city.iloc[2].Longitude),
                        city=each.AccentCity,
                    )

            elif p_type == '1':
                prompt = template[3].format(
                        city_a=sampled_city.iloc[0].AccentCity,
                        lat_a=str(sampled_city.iloc[0].Latitude),
                        lng_a=str(sampled_city.iloc[0].Longitude),
                        city_a=sampled_city.iloc[1].AccentCity,
                        lat_a=str(sampled_city.iloc[1].Latitude),
                        lng_a=str(sampled_city.iloc[1].Longitude),
                        city_a=sampled_city.iloc[2].AccentCity,
                        lat_a=str(sampled_city.iloc[2].Latitude),
                        lng_a=str(sampled_city.iloc[2].Longitude),
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
            'model_size': model_name.split('/')[1].split('-')[1]
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
            'model_size': model_name.split('/')[1].split('-')[1]
        })
        results.append(res)
    return results
