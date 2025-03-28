# coding=utf-8
# Created : 2023/2/28 17:59
# Author  : Zy
import clip
import torch
from PIL import Image
import skimage.io as io
import onnxruntime as ort
import torch.nn.functional as F
from transformers import BertTokenizer
from common.utils import Util


class ClipCapChinese:
    def __init__(self):
        super(ClipCapChinese, self).__init__()
        self.root_path = Util.app_path() + "/models/vision/clip_gpt/"
        self.__gpt2_model_path = self.root_path + "gpt2"
        self.__model_path = self.root_path + "gpt.pth"
        self.__clip_model_path = self.root_path + "ViT-B-32.pt"
        self.__mlp_model_path = self.root_path + "clip_project.onnx"
        self.cuda = torch.cuda.is_available()
        self.__gpt2 = torch.load(self.__model_path)
        self.__gpt2 = self.__gpt2 if not self.cuda else self.__gpt2.cuda()
        self.__tokenizer = BertTokenizer.from_pretrained(self.__gpt2_model_path)
        self.__clip_model, self.preprocess = clip.load(self.__clip_model_path, jit=False)
        self.__clip_project = ort.InferenceSession(self.__mlp_model_path)

    def __topk_filtering(self, logits, topp=.0, filter_value=-float('Inf')):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > topp
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # todo check
        for i in range(sorted_indices_to_remove.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        return logits

    def __generate(self, clip_embeds, tokenizer):
        b_size = clip_embeds.size(0)
        pad_id = tokenizer.pad_token_id
        sep_id = tokenizer.sep_token_id
        unk_id = tokenizer.unk_token_id
        max_len = 100
        topp = 0.8

        cur_len = 0
        caption_ids = []

        clip_embeds = clip_embeds.cpu().detach().numpy().astype("float32")
        inputs_embeds = self.__clip_project.run(None, {"input.1": clip_embeds})[0].reshape(-1, 10, 768)
        inputs_embeds = torch.LongTensor(inputs_embeds) if not self.cuda else torch.LongTensor(inputs_embeds).cuda()
        finish_flag = [False] * b_size
        while True:
            logits = self.__gpt2(inputs_embeds=inputs_embeds).logits
            next_token_logits = logits[:, -1, :]
            next_token_logits[:, unk_id] = -float('Inf')

            filtered_logits = self.__topk_filtering(next_token_logits, topp)
            next_token_ids = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(1).tolist()

            for index in range(len(next_token_ids)):
                token_id = next_token_ids[index]
                if finish_flag[index]:
                    next_token_ids[index] = pad_id
                elif token_id == sep_id:
                    finish_flag[index] = True
                elif cur_len == 0:
                    caption_ids.append([token_id])
                else:
                    caption_ids[index].append(token_id)
            next_token_ids = torch.tensor(next_token_ids)
            next_token_ids = next_token_ids.cuda() if self.cuda else next_token_ids
            next_token_embeds = self.__gpt2.transformer.wte(next_token_ids).unsqueeze(1)
            next_token_embeds = next_token_embeds.cuda() if self.cuda else next_token_embeds
            inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)
            cur_len += 1
            if cur_len > max_len or False not in finish_flag:
                break

        caption = tokenizer.convert_ids_to_tokens(caption_ids[0])
        caption = ''.join(caption)
        return caption

    def predict(self, filename: str) -> str:
        image = io.imread(filename)
        proc_image = self.preprocess(Image.fromarray(image))
        proc_image = proc_image.unsqueeze(0)
        clip_embeds = self.__clip_model.encode_image(proc_image)
        clip_embeds = clip_embeds.unsqueeze(1).view(-1, clip_embeds.size(-1))
        caption = self.__generate(clip_embeds, self.__tokenizer)
        return caption
