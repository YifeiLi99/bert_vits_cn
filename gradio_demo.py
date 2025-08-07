import numpy as np
import torch
import gradio as gr
from my_utils import runtime_utils,checkpoint_utils,config_utils
from text.symbols import symbols
from text.sequence import cleaned_text_to_sequence
from frontend.pinyin_frontend import VITS_PinYin

# ====== 预加载部分 =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts_front = VITS_PinYin("./bert", device)
hps = config_utils.get_hparams("config.yaml")

net_g = runtime_utils.load_class(hps.train.eval_class)(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
checkpoint_utils.load_model("vits_bert_model.pth", net_g)
net_g.eval().to(device)

# ====== 推理函数 =======
def infer_text(text):
    phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
    input_ids = cleaned_text_to_sequence(phonemes)
    with torch.no_grad():
        x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
        x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.666, length_scale=1)[0][0, 0].data.cpu().float().numpy()
    # 归一化后返回Gradio播放
    audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
    return hps.data.sampling_rate, audio.astype(np.int16)

# ====== Gradio界面 =======
iface = gr.Interface(fn=infer_text,
                     inputs=gr.Textbox(lines=2, placeholder="输入中文文本..."),
                     outputs="audio",
                     title="BERT-VITS 语音合成 Demo")

if __name__ == "__main__":
    iface.launch()
