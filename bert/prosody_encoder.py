import os
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer

# ================================
# 基于BERT的韵律特征提取模块
# 1. 将输入文本转为字符embedding
# 2. 输出韵律向量序列 (如重音、节奏、时长等特征)
# ================================

class CharEmbedding(nn.Module):
    """
    基于BERT的字符级Embedding模型
    - 输入：文本（转为token id序列）
    - 输出：降维后的embedding序列（256维 -> 3维韵律向量）
    """
    def __init__(self, model_dir):
        super().__init__()
        # 加载BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        # 加载BERT配置与模型
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.bert = BertModel(self.bert_config)
        # 降维映射层，将BERT输出降到256维
        self.proj = nn.Linear(self.hidden_size, 256)
        # 最终输出韵律特征（3维，可理解为节奏/重音/时长）
        self.linear = nn.Linear(256, 3)

    def text2Token(self, text):
        """
        将文本转为BERT token id序列
        """
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        """
        前向传播：
        - 输入BERT token id序列及mask
        - 输出降维后的字符embedding序列
        """
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)  # 降维到256维
        return out_seq


class TTSProsody:
    """
    韵律特征提取器（封装BERT模型及推理流程）
    - 支持加载预训练权重
    - 输入文本 -> 输出韵律向量序列
    """
    def __init__(self, path, device):
        self.device = device
        self.char_model = CharEmbedding(path)
        # 加载 prosody_model.pt 权重文件
        self.char_model.load_state_dict(
            torch.load(
                os.path.join(path, 'prosody_model.pt'),
                map_location="cpu"
            ),
            strict=False  # 忽略不匹配的参数
        )
        self.char_model.eval()
        self.char_model.to(self.device)

    def get_char_embeds(self, text):
        """
        将输入文本转为韵律embedding序列
        Returns:
            Tensor: [chars, 256] embedding序列
        """
        input_ids = self.char_model.text2Token(text)
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)

        # 转为Tensor并送入设备
        input_ids = torch.LongTensor([input_ids]).to(self.device)
        input_masks = torch.LongTensor([input_masks]).to(self.device)
        type_ids = torch.LongTensor([type_ids]).to(self.device)

        # 推理（禁用梯度）
        with torch.no_grad():
            char_embeds = self.char_model(
                input_ids, input_masks, type_ids).squeeze(0).cpu()
        return char_embeds

    def expand_for_phone(self, char_embeds, length):
        """
        将字符embedding扩展到音素级别（用于对齐TTS音素序列）
        Args:
            char_embeds (Tensor): [chars, 256] 字符embedding序列
            length (List[int]): 每个字符对应的音素数
        Returns:
            np.ndarray: [phones, 256] 扩展后的embedding序列
        """
        assert char_embeds.size(0) == len(length)

        expand_vecs = []
        for vec, leng in zip(char_embeds, length):
            vec = vec.expand(leng, -1)  # 将字符embedding扩展到对应音素数
            expand_vecs.append(vec)

        expand_embeds = torch.cat(expand_vecs, dim=0)
        assert expand_embeds.size(0) == sum(length)
        return expand_embeds.numpy()
