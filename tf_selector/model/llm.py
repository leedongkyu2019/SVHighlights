import torch
import transformers

import torch.nn as nn

from transformers import AutoTokenizer


def get_model(args):
    model_name, temperature, max_new_tokens = args.model, args.temperature, args.max_new_tokens
    if 'Llama-2' in model_name:
        return LLaMA2(model_name, temperature, max_new_tokens)
    elif 'Llama-3' in model_name:
        return LLaMA3(model_name, temperature, max_new_tokens)

class LLaMA2(nn.Module):
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            tokenizer=tokenizer,
            temperature=temperature
        )

    def forward(self, head, prompt):
        message = [
            {"role": "user", "content": prompt}
        ]
        info = {}
        sequences = self.pipeline(
            message,
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        response = sequences[0]['generated_text'][-1]["content"]
        info = {
            'message': prompt,
            'response': response
        }
        return response, info

class LLaMA3(nn.Module):
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=model_name, 
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def forward(self, head, prompt):
        message = [
            {"role": "user", "content": prompt}
        ]
        info = {}
        if 'Llama-3-' in self.model_name: # llama-3
            sequences = self.pipeline(
                message,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.terminators,
                do_sample=False,
                temperature=self.temperature,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )
        elif 'Llama-3.1' in self.model_name: # llama-3.1
            sequences = self.pipeline(
                message,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )
        response = sequences[0]["generated_text"][-1]["content"]
        info = {
            'message': prompt,
            'response': response
        }
        return response, info