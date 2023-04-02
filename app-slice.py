import os
import gradio as gr
import librosa
import numpy as np
from pathlib import Path
import inference.infer_tool as infer_tool
import utils
from inference.infer_tool import Svc
import logging
import webbrowser
import argparse
import soundfile
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
    def vc_fn(input_audio, vc_transform, auto_f0, slice_db, noise_scale, pad_seconds):
        if input_audio is None:
            return "You need to select an audio", None
        raw_audio_path = f"raw/{input_audio}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        _audio = model.slice_inference(
            wav_path, sid, vc_transform, slice_db,
            cluster_infer_ratio=0,
            auto_predict_f0=auto_f0,
            noice_scale=noise_scale,
            pad_seconds=pad_seconds)
        model.clear_empty()
        return "Success", (44100, _audio)
    return vc_fn

def refresh_raw_wav():
    return gr.Dropdown.update(choices=os.listdir("raw"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--colab", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    hubert_model = utils.get_hubert_model().to(args.device)
    models = []
    raw = os.listdir("raw")
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
                            with gr.Row():
                                vc_input = gr.Dropdown(label="Input audio", choices=raw)
                                vc_refresh = gr.Button("🔁", variant="primary")
                            vc_transform = gr.Number(label="vc_transform", value=0)
                            slice_db = gr.Number(label="slice_db", value=-40)
                            noise_scale = gr.Number(label="noise_scale", value=0.4)
                            pad_seconds = gr.Number(label="pad_seconds", value=0.5)
                            auto_f0 = gr.Checkbox(label="auto_f0", value=False)
                            vc_submit = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            vc_output1 = gr.Textbox(label="Output Message")
                            vc_output2 = gr.Audio(label="Output Audio")
                vc_submit.click(vc_fn, [vc_input, vc_transform, auto_f0, slice_db,  noise_scale, pad_seconds], [vc_output1, vc_output2])
                vc_refresh.click(refresh_raw_wav, [], [vc_input])
        if args.colab:
            webbrowser.open("http://127.0.0.1:7860")
        app.queue(concurrency_count=1, api_open=args.api).launch(share=args.share)