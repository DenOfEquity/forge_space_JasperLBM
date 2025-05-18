from copy import deepcopy

import gradio as gr
import PIL
import spaces
import torch
import gc
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoModelForImageSegmentation
from utils import extract_object, get_model_from_config, resize_and_center_crop


ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}

# download the config and model
MODEL_PATH = hf_hub_download("jasperai/LBM_relighting", "model.safetensors")
CONFIG_PATH = hf_hub_download("jasperai/LBM_relighting", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
model = get_model_from_config(**config)
sd = load_file(MODEL_PATH)
model.load_state_dict(sd, strict=True)
# model.to(torch.bfloat16)
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet_HR", trust_remote_code=True
)
birefnet.eval()
birefnet.half()

model.vae.to(torch.float32)
model.denoiser.to(torch.bfloat16)

spaces.change_attention_from_diffusers_to_forge(model.vae.vae_model)

def evaluate(
    fg_image: PIL.Image.Image,
    bg_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):

    ori_h_bg, ori_w_bg = fg_image.size
    ar_bg = ori_h_bg / ori_w_bg
    closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    dimensions_bg = ASPECT_RATIOS[closest_ar_bg]

    birefnet.cuda()
    _, fg_mask = extract_object(birefnet, deepcopy(fg_image))
    birefnet.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    fg_image = resize_and_center_crop(fg_image, dimensions_bg[0], dimensions_bg[1])
    fg_mask = resize_and_center_crop(fg_mask, dimensions_bg[0], dimensions_bg[1])
    bg_image = resize_and_center_crop(bg_image, dimensions_bg[0], dimensions_bg[1])

    img_pasted = PIL.Image.composite(fg_image, bg_image, fg_mask)

    img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }

    model.vae.cuda()
    z_source = model.vae.encode(img_pasted_tensor.cuda().float())
    model.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    model.denoiser.cuda()
    output_latent = model.sample(
        z=z_source.to(torch.bfloat16),
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    )
    model.denoiser.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    model.vae.cuda()
    output_image = model.vae.decode(output_latent.float()).clamp(-1, 1)
    model.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)

    # paste the output image on the background image
    output_image = PIL.Image.composite(output_image, bg_image, fg_mask)

    output_image.resize((ori_h_bg, ori_w_bg))

    return output_image


def unload():
    global model, birefnet
    del model, birefnet
    gc.collect()
    torch.cuda.empty_cache()


css = """
footer {
    display: none !important;
}
"""


with gr.Blocks(analytics_enabled=False, css=css, title="LBM Object Relighting") as demo:
    gr.Markdown(
        """
        # Object Relighting
        ## [LBM: Latent Bridge Matching for Fast Image-to-Image Translation](https://arxiv.org/abs/2503.07535) *by Jasper Research*.
    """
    )

# adjust positioning of foreground? - expand foreground

    with gr.Row():
        with gr.Column():
            with gr.Group():
                fg_image = gr.Image(
                    type="pil",
                    label="Input image",
                    image_mode="RGB", sources=['upload', 'clipboard'],
                    height=360,
                )

                bg_image = gr.Image(
                    type="pil",
                    label="Target background",
                    image_mode="RGB", sources=['upload', 'clipboard'],
                    height=360,
                )

                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=1,
                    step=1,
                    label="Number of inference steps",
                )

        with gr.Column():
            submit_button = gr.Button("Relight", variant="primary")
            output_slider = gr.Image(label="Result", type="pil", height=733)


    def submitOff():
        return gr.Button("... working ...", variant="secondary")
    def submitOn():
        return gr.Button("Relight", variant="primary")

    submit_button.click(fn=submitOff, outputs=submit_button).then(
        evaluate,
        inputs=[fg_image, bg_image, num_inference_steps],
        outputs=[output_slider],
        show_progress="full",
        show_api=False,
    ).then(fn=submitOn, outputs=submit_button)


    gr.Markdown(
        """
        If you enjoy the space, please also promote *open-source* by giving a ⭐ to the <a href='https://github.com/gojasper/LBM' target='_blank'>Github Repo</a>.
        """
    )

    gr.Markdown("**Disclaimer:**")
    gr.Markdown(
        "This demo is only for research purpose. Jasper cannot be held responsible for the generation of NSFW (Not Safe For Work) content through the use of this demo. Users are solely responsible for any content they create, and it is their obligation to ensure that it adheres to appropriate and ethical standards. Jasper provides the tools, but the responsibility for their use lies with the individual user."
    )

    demo.unload(fn=unload)

if __name__ == "__main__":
    demo.queue().launch(show_api=False)

