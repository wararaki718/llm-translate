from functools import partial

import transformers
import torch


class LLMTranslator:
    def __init__(self, model_id: str="meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda"
        )
        self._convert = partial(
            pipeline.tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        self._generate = partial(
            pipeline,
            max_new_tokens=256,
            eos_token_id=[
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    def translate(self, text: str) -> str:
        messages = {
            {"role": "system", "content": "日本語で会話してください。"},
            {"role": "user", "content": text},
        }
        prompt = self._convert(messages)
        outputs = self._generate(prompt)
        result: str = outputs[0]["generated_text"][len(prompt):]
        return result
