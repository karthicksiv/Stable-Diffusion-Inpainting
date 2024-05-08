# Stable-Diffusion-Inpainting

**Stable Diffusion Inpainting** is a project that combines the power of Stable Diffusion and the Segment Anything Model (SAM) to generate seamless backgrounds for images. By simply clicking on the main item in a picture, users can automatically generate a background that blends perfectly with the existing image.

## Features
- Intuitive user interface built with Gradio for selecting the main item in an image.
- Utilizes Stable Diffusion for high-quality image inpainting.
- Leverages SAM for precise object segmentation based on user clicks.
- Generates backgrounds that seamlessly blend with the original image.
- Supports custom prompts for guided image generation.
- Includes a comprehensive list of negative prompts to refine the output quality.

## Installation

1. **Clone the repository:**
   
``` bash
git clone https://github.com/yourusername/Stable-Diffusion-Inpainting.git
```


2. **Install the required dependencies:**

``` bash
cd Stable-Diffusion-Inpainting
pip install -r requirements.txt
```


3. **Download the necessary model checkpoints:**
- SAM checkpoint: [link_to_sam_checkpoint]
- Place the downloaded checkpoint in the `weights` directory.

## Usage

1. **Run the application:**

``` bash
python app.py
```


2. Open a web browser and navigate to the provided URL (e.g., http://localhost:7860).
3. Upload an image and click on the main item in the picture.
4. The application will generate a mask based on your clicks and display it in the "Mask" section.
5. Enter a prompt to guide the background generation process.
6. Click the "Submit" button to generate a seamless background that blends with the original image.
7. The generated output will be displayed in the "Output" section.

## Customization

- You can modify the `get_negatives()` function to customize the list of negative prompts used for refining the output quality.
- Experiment with different prompts to achieve the desired background generation results.
- Adjust the image resizing logic in the `resize_image_preserving_aspect_ratio()` function if needed.

## Dependencies

- **Gradio**: For building the user interface
- **Stable Diffusion**: For image inpainting
- **Segment Anything Model (SAM)**: For object segmentation
- **PyTorch**: For deep learning computations
- **Diffusers**: For accessing pre-trained Stable Diffusion models
- **Accelerate**: For accelerated training and inference
- **Pillow**: For image processing

