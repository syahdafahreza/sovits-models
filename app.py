import io
import os
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_vc_fn(model, sid):
    def vc_fn(input_audio, vc_transform, auto_f0):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        duration = audio.shape[0] / sampling_rate
        if duration > 45:
            return "Please upload an audio file that is less than 45 seconds. If you need to generate a longer audio file, please use Colab.", None
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        out_wav_path = "temp.wav"
        soundfile.write(out_wav_path, audio, 16000, format="wav")
        out_audio, out_sr = model.infer(sid, vc_transform, out_wav_path,
                                       auto_predict_f0=auto_f0,
                                       )
        return "Success", (44100, out_audio.cpu().numpy())
    return vc_fn

if __name__ == '__main__':
    models = []
    for f in os.listdir("models"):
        name = f
        model = Svc(fr"models/{f}/{f}.pth", f"models/{f}/config.json")
        cover = f"models/{f}/cover.png" if os.path.exists(f"models/{f}/cover.png") else None
        models.append((name, cover, create_vc_fn(model, name)))
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> Sovits Umamusume\n"
            "## <center> The input audio should be clean and pure voice without background music.\n"
            "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=sayashi.Sovits-Umamusume)\n\n"
            "[Open In Colab](https://colab.research.google.com/drive/1wfsBbMzmtLflOJeqc5ZnJiLY7L239hJW?usp=share_link)"
            "\n\n"
            "[Original Repo](https://github.com/innnky/so-vits-svc/tree/4.0)"
        )
        with gr.Tabs():
            for (name, cover, vc_fn) in models:
                with gr.TabItem(name):
                    with gr.Row():
                        gr.Markdown(
                            '<div align="center">'
                            f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else ""
                            '</div>'
                        )
                    with gr.Row():
                        with gr.Column():
                            vc_input = gr.Audio(label="Input audio (less than 45 seconds)")
                            vc_transform = gr.Number(label="vc_transform", value=0)
                            auto_f0 = gr.Checkbox(label="auto_f0", value=False)
                            vc_submit = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            vc_output1 = gr.Textbox(label="Output Message")
                            vc_output2 = gr.Audio(label="Output Audio")
                vc_submit.click(vc_fn, [vc_input, vc_transform, auto_f0], [vc_output1, vc_output2])
        app.queue(concurrency_count=1).launch()



