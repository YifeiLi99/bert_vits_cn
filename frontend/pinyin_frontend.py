from pypinyin import lazy_pinyin, Style
from pypinyin.core import load_phrases_dict  # 动态加载自定义词典拼音
from text.pinyin_dict import pinyin_dict  # 音素映射表
from bert.prosody_encoder import TTSProsody  # BERT韵律特征提取模块

# 判断字符是否为汉字
def is_chinese(uchar):
    return u'\u4e00' <= uchar <= u'\u9fa5'

# 清理文本，非中文字符替换成逗号（作为停顿标记）
def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if is_chinese(char):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')  # 非中文用逗号作为停顿符
    text_clean = ''.join(text_clean).strip(',')
    return text_clean

# 加载本地拼音字典（覆盖pypinyin默认词库）
def load_pinyin_dict():
    my_dict = {}
    with open("./text/pinyin-local.txt", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            phone = [[p] for p in cuts[1:]]  # 形如 {"你": [["n"], ["i3"]]}
            my_dict[hanzi] = phone
    load_phrases_dict(my_dict)  # 动态加载自定义拼音词典

# =============================================
# VITS_PinYin 类：负责汉字转拼音 + 韵律特征提取
# =============================================
class VITS_PinYin:
    def __init__(self, bert_path, device, hasBert=True):
        load_pinyin_dict()  # 初始化加载拼音字典
        self.hasBert = hasBert
        if self.hasBert:
            self.prosody = TTSProsody(bert_path, device)  # 初始化BERT韵律模型
        # self.normalizer = Normalizer()  # 文本规约（未启用）

    # 将拼音转为音素序列（如 "ni3" → "n", "i3"）
    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []  # 记录每个字占用的音素数
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]  # 声调数字
                base = pinyin[:-1]  # 不带声调部分
                a1, a2 = pinyin_dict[base]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    # 主流程：中文句子 → 音素序列 + (可选) BERT韵律特征
    def chinese_to_phonemes(self, text):
        # text = self.normalizer.normalize(text)  # 可选文本规约（暂未启用）
        text = clean_chinese(text)
        phonemes = ["sil"]  # 句首静音标记
        chars = ['[PAD]']  # BERT输入字符序列
        count_phone = [1]  # 对应 [PAD] 占用1个音素
        for subtext in text.split(","):  # 按逗号分段
            if len(subtext) == 0:
                continue
            pinyins = self.correct_pinyin_tone3(subtext)  # 拼音标注 (Tone3格式)
            sub_p, sub_c = self.get_phoneme4pinyin(pinyins)  # 转音素
            phonemes.extend(sub_p)
            phonemes.append("sp")  # 子句间停顿
            count_phone.extend(sub_c)
            count_phone.append(1)
            chars.append(subtext)
            chars.append(',')
        phonemes.append("sil")  # 句尾静音
        count_phone.append(1)
        chars.append('[PAD]')
        chars = "".join(chars)

        char_embeds = None
        if self.hasBert:
            # BERT 获取每个字符的embedding向量
            char_embeds = self.prosody.get_char_embeds(chars)
            # 将字符embedding扩展到每个音素上
            char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return " ".join(phonemes), char_embeds

    # 调用pypinyin的Tone3风格拼音标注（带声调数字）
    def correct_pinyin_tone3(self, text):
        pinyin_list = lazy_pinyin(
            text,
            style=Style.TONE3,
            strict=False,
            neutral_tone_with_five=True,
            tone_sandhi=True  # 三声变调
        )
        return pinyin_list
