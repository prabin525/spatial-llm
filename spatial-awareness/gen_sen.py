def gen_sen(
        model,
        tokenizer,
        device,
        cities,
        p_type='near',
        p_length='zero-shot',
        state=True
):

    results = []
    template = [
        open('templates/near-zero-shot.txt').read(),
        open('templates/and-zero-shot.txt').read(),
        open('templates/far-zero-shot.txt').read(),
        open('templates/close-zero-shot.txt').read(),
        open('templates/near-3-shot.txt').read(),
        open('templates/and-3-shot.txt').read(),
        open('templates/far-3-shot.txt').read(),
        open('templates/close-3-shot.txt').read(),
    ]

    for i, each in cities.iterrows():
        res = each.to_dict()
        if p_length == 'zero-shot':
            if p_type == 'near':
                if state:
                    prompt = template[0].format(
                            city=each["name"],
                        )
                else:
                    prompt = template[0].format(
                            city=each["name"].split(",")[0],
                        )

            elif p_type == 'and':
                if state:
                    prompt = template[1].format(
                            city=each["name"],
                        )
                else:
                    prompt = template[1].format(
                            city=each["name"].split(",")[0],
                        )
            elif p_type == 'far':
                if state:
                    prompt = template[2].format(
                            city=each["name"],
                        )
                else:
                    prompt = template[2].format(
                            city=each["name"].split(",")[0],
                        )
            elif p_type == 'close':
                if state:
                    prompt = template[3].format(
                            city=each["name"],
                        )
                else:
                    prompt = template[3].format(
                            city=each["name"].split(",")[0],
                        )
        elif p_length == '3-shot':
            remaining = cities.loc[cities.index != i]
            if p_type == 'near':
                if state:
                    c = remaining.sample(3)
                    prompt = template[4].format(
                            city_a=c.iloc[0]["name"],
                            city_b=c.iloc[0].near_city,
                            city_c=c.iloc[1]["name"],
                            city_d=c.iloc[1].near_city,
                            city_e=c.iloc[2]["name"],
                            city_f=c.iloc[2].near_city,
                            city=each["name"],
                        )
                else:
                    c = remaining.sample(3)
                    prompt = template[4].format(
                            city_a=c.iloc[0]["name"].split(",")[0],
                            city_b=c.iloc[0].near_city.split(",")[0],
                            city_c=c.iloc[1]["name"].split(",")[0],
                            city_d=c.iloc[1].near_city.split(",")[0],
                            city_e=c.iloc[2]["name"].split(",")[0],
                            city_f=c.iloc[2].near_city.split(",")[0],
                            city=each["name"].split(",")[0],
                        )

            elif p_type == 'and':
                if state:
                    c = remaining.sample(6)
                    prompt = template[5].format(
                            city_a=c.iloc[0]["name"],
                            city_b=c.iloc[1]["name"],
                            city_c=c.iloc[2]["name"],
                            city_d=c.iloc[3]["name"],
                            city_e=c.iloc[4]["name"],
                            city_f=c.iloc[5]["name"],
                            city=each["name"],
                        )
                else:
                    c = remaining.sample(3)
                    prompt = template[5].format(
                            city_a=c.iloc[0]["name"].split(",")[0],
                            city_b=c.iloc[1]["name"].split(",")[0],
                            city_c=c.iloc[2]["name"].split(",")[0],
                            city_d=c.iloc[3]["name"].split(",")[0],
                            city_e=c.iloc[4]["name"].split(",")[0],
                            city_f=c.iloc[5]["name"].split(",")[0],
                            city=each["name"],
                        )
            elif p_type == 'far':
                if state:
                    c = remaining.sample(3)
                    prompt = template[6].format(
                            city_a=c.iloc[0]["name"],
                            city_b=c.iloc[0].far_city,
                            city_c=c.iloc[1]["name"],
                            city_d=c.iloc[1].far_city,
                            city_e=c.iloc[2]["name"],
                            city_f=c.iloc[2].far_city,
                            city=each["name"],
                        )
                else:
                    c = remaining.sample(3)
                    prompt = template[6].format(
                            city_a=c.iloc[0]["name"].split(",")[0],
                            city_b=c.iloc[0].far_city.split(",")[0],
                            city_c=c.iloc[1]["name"].split(",")[0],
                            city_d=c.iloc[1].far_city.split(",")[0],
                            city_e=c.iloc[2]["name"].split(",")[0],
                            city_f=c.iloc[2].far_city.split(",")[0],
                            city=each["name"].split(",")[0],
                        )
            elif p_type == 'close':
                if state:
                    c = remaining.sample(3)
                    prompt = template[7].format(
                            city_a=c.iloc[0]["name"],
                            city_b=c.iloc[0].near_city,
                            city_c=c.iloc[1]["name"],
                            city_d=c.iloc[1].near_city,
                            city_e=c.iloc[2]["name"],
                            city_f=c.iloc[2].near_city,
                            city=each["name"],
                        )
                else:
                    c = remaining.sample(3)
                    prompt = template[7].format(
                            city_a=c.iloc[0]["name"].split(",")[0],
                            city_b=c.iloc[0].near_city.split(",")[0],
                            city_c=c.iloc[1]["name"].split(",")[0],
                            city_d=c.iloc[1].near_city.split(",")[0],
                            city_e=c.iloc[2]["name"].split(",")[0],
                            city_f=c.iloc[2].near_city.split(",")[0],
                            city=each["name"].split(",")[0],
                        )

        tokenized_prompt = tokenizer.encode(
            prompt,
            return_tensors='pt'
        )
        tokenized_prompt = tokenized_prompt.to(device)
        gen = model.generate(
            tokenized_prompt,
            do_sample=True,
            top_k=100,
            temperature=0.9,
            max_new_tokens=10,
            num_return_sequences=50
            # eos_token_id=tokenizer.encode('\n\n')
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        res.update({
            'output': outputs,
            'prompt': prompt,
            'p_type': p_type,
            'p_length': p_length,
            'state': 'state' if state else 'no-state'
        })
        results.append(res)
        break
    return results
