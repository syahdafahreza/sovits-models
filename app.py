import os
import io
import gradio as gr
import librosa
import numpy as np
import utils
from inference.infer_tool import Svc
import logging
import soundfile
import argparse
import gradio.processing_utils as gr_processing_utils
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"  # limit audio length in huggingface spaces

audio_postprocess_ori = gr.Audio.postprocess

def audio_postprocess(self, y):
    data = audio_postprocess_ori(self, y)
    if data is None:
        return None
    return gr_processing_utils.encode_url_or_file_to_base64(data["name"])


gr.Audio.postprocess = audio_postprocess
def create_vc_fn(model, sid):
    def vc_fn(input_audio, vc_transform, auto_f0):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        if duration > 20 and limitation:
            return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        raw_path = io.BytesIO()
        soundfile.write(raw_path, audio, 16000, format="wav")
        raw_path.seek(0)
        out_audio, out_sr = model.infer(sid, vc_transform, raw_path,
                                       auto_predict_f0=auto_f0,
                                       )
        return "Success", (44100, out_audio.cpu().numpy())
    return vc_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    hubert_model = utils.get_hubert_model().to(args.device)
    models = []
    for f in os.listdir("models"):
        name = f
        model = Svc(fr"models/{f}/{f}.pth", f"models/{f}/config.json", device=args.device, hubert_model=hubert_model)
        cover = f"models/{f}/cover.png" if os.path.exists(f"models/{f}/cover.png") else None
        models.append((name, cover, create_vc_fn(model, name)))
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> Sovits Models\n"
            "## <center> The input audio should be clean and pure voice without background music.\n"
            "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=sayashi.Sovits-Umamusume)\n\n"
            "[Open In Colab](https://colab.research.google.com/drive/1wfsBbMzmtLflOJeqc5ZnJiLY7L239hJW?usp=share_link)"
            " without queue and length limitation.\n\n"
            "[Original Repo](https://github.com/svc-develop-team/so-vits-svc)\n\n"
            "Other models:\n"
            "[rudolf](https://huggingface.co/spaces/sayashi/sovits-rudolf)\n"
            "[teio](https://huggingface.co/spaces/sayashi/sovits-teio)\n"
            "[goldship](https://huggingface.co/spaces/sayashi/sovits-goldship)\n"
            "[tannhauser](https://huggingface.co/spaces/sayashi/sovits-tannhauser)\n"

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
                            vc_input = gr.Audio(label="Input audio"+' (less than 20 seconds)' if limitation else '')
                            vc_transform = gr.Number(label="vc_transform", value=0)
                            auto_f0 = gr.Checkbox(label="auto_f0", value=False)
                            vc_submit = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            vc_output1 = gr.Textbox(label="Output Message")
                            vc_output2 = gr.Audio(label="Output Audio")
                vc_submit.click(vc_fn, [vc_input, vc_transform, auto_f0], [vc_output1, vc_output2])
        app.queue(concurrency_count=1, api_open=args.api).launch(share=args.share)