import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

def resize_image_preserving_aspect_ratio(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    return resized_image

accelerator = Accelerator()

device = 'cpu'
# sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
sam_checkpoint = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
model_type = 'vit_h'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16"
    # "stabilityai/stable-diffusion-2-inpainting",
    # custom_pipeline = 'lpw_stable_diffusion'
    # torch_dtype=torch.float16
)
device = accelerator.device

pipe = pipe.to(device)

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label='Input')
        mask_img = gr.Image(label='Mask')
        output_img = gr.Image(label='Output')

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label='Prompt')
    
    with gr.Row():
        submit = gr.Button('Submit')

    def generate_mask(image, evt: gr.SelectData):
        selected_pixels.append(evt.index)

        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        mask, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        # (n, sz, sz)
        mask = np.logical_not(mask)
        mask = Image.fromarray(mask[0, : , :].astype(np.uint8) * 255)
        return mask
    
    def get_negatives():
        # Enhanced list of negatives including additional terms for more refined output
        negatives = """
        lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, 
        bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, 
        worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry, deformed iris, deformed pupils, 
        semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, 
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, 
        disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, 
        long neck, watermark, logo
        """
        return negatives

    

    def inpaint(image, mask, prompt):
        original_image = Image.fromarray(image).convert("RGB")
        original_size = original_image.size

        # Resize input image preserving aspect ratio to fit the model's expected size
        resized_image = resize_image_preserving_aspect_ratio(original_image, (512, 512))
        resized_mask = resize_image_preserving_aspect_ratio(Image.fromarray(mask), (512, 512))

        # Process with the model
        output = pipe(prompt=prompt, negative_prompt=get_negatives(), image=resized_image, mask_image=resized_mask).images[0]

        # Resize output back to original dimensions
        output_resized_to_original = output.resize(original_size, Image.Resampling.LANCZOS)

        return output_resized_to_original
    
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(
        inpaint, 
        inputs=[input_img, mask_img, prompt_text],
        outputs=[output_img],
    )

if __name__ == "__main__":
    demo.launch()