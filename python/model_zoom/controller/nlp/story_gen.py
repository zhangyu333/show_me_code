# coding=utf-8
# Created : 2023/1/11 14:21
# Author  : Zy
import torch
import numpy as np
import onnxruntime as ort
from transformers import CpmTokenizer
from common.utils import Util


class StoryGenerate:
    def __init__(self):
        super(StoryGenerate, self).__init__()
        self.tokenizer = CpmTokenizer(vocab_file=Util.app_path() + "/static/vocab/chinese_vocab.model")
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<eod>")
        self.sep_id = self.tokenizer.sep_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.model = ort.InferenceSession(Util.app_path() + "/models/nlg/GPT2LMHeadModelINT8.onnx")
        self.input_name = [self.model.get_outputs()[0].name]
        self.output_name = self.model.get_inputs()[0].name

    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax

    def topKTopPFiltering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        assert logits.dim() == 1
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.Tensor(self.softmax(sorted_logits.numpy())), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def generateNextToken(self, input_ids):
        pad_input_ids = input_ids.numpy().astype("int64")
        if 100 - input_ids.shape[1] > 0:
            pad = np.zeros((1, 100 - input_ids.shape[1]), dtype=np.int64)
            pad_input_ids = np.hstack((pad_input_ids, pad))
        outputs = self.model.run(self.input_name, {self.output_name: pad_input_ids})
        logits = outputs[0]
        logits = logits[:, :input_ids.shape[1], :]
        logits = torch.Tensor(logits)
        next_token_logits = logits[0, -1, :]
        next_token_logits[self.unk_id] = -float('Inf')
        filtered_logits = self.topKTopPFiltering(next_token_logits, top_k=0, top_p=0.85)
        next_token_id = torch.multinomial(torch.Tensor(self.softmax(filtered_logits.numpy())), num_samples=1)
        return next_token_id

    def generate(self, title: str, context: str):
        title_ids = self.tokenizer.encode(title, add_special_tokens=False)
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        input_ids = title_ids + [self.sep_id] + context_ids
        cur_len = len(input_ids)
        last_token_id = input_ids[-1]
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        while True:
            next_token_id = self.generateNextToken(input_ids[:, -100:])
            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
            cur_len += 1
            word = self.tokenizer.convert_ids_to_tokens(next_token_id.item())
            if cur_len >= 100 and last_token_id == 8 and next_token_id == 3:
                break
            if cur_len >= 100 and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
                break
            if next_token_id == self.eod_id:
                break
        result = self.tokenizer.decode(input_ids.squeeze(0))
        result = result.split("<sep>")[1].replace("\n", "").replace(" ", "")
        result = result[:result.rfind("。")+1]
        return result


if __name__ == '__main__':
    pass
