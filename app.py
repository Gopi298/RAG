import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Text to Image Generator",
    page_icon="🎨",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎨 AI Text-to-Image Generator")
st.write(
    "Enter a text prompt and generate an AI image using a "
    "Stable Diffusion model."
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
    st.success("GPU detected - CUDA will be used.")
else:
    device = "cpu"
    st.warning(
        "GPU not detected. Image generation will use CPU "
        "and may be slow."
    )

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Generation Settings")

width = st.sidebar.selectbox(
    "Image Width",
    [256, 384, 512, 640, 768],
    index=2
)

height = st.sidebar.selectbox(
    "Image Height",
    [256, 384, 512, 640, 768],
    index=2
)

steps = st.sidebar.slider(
    "Inference Steps",
    min_value=10,
    max_value=100,
    value=30,
    step=5
)

guidance = st.sidebar.slider(
    "Guidance Scale",
    min_value=1.0,
    max_value=20.0,
    value=7.5,
    step=0.5
)

seed_option = st.sidebar.selectbox(
    "Seed",
    [
        "Random",
        "Fixed"
    ]
)

if seed_option == "Fixed":

    seed = st.sidebar.number_input(
        "Enter Seed",
        min_value=0,
        max_value=999999999,
        value=42
    )

else:

    seed = random.randint(0, 999999999)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = st.text_area(
    "✍️ Enter your prompt",
    placeholder=(
        "Example: A futuristic Chennai city at night, "
        "cinematic lighting, highly detailed, realistic"
    ),
    height=120
)

# --------------------------------------------------
# NEGATIVE PROMPT
# --------------------------------------------------

negative_prompt = st.text_area(
    "🚫 Negative Prompt",
    value=(
        "blurry, low quality, distorted, deformed, "
        "bad anatomy, extra fingers, extra limbs, "
        "duplicate, watermark, text"
    ),
    height=100
)

# --------------------------------------------------
# GENERATE BUTTON
# --------------------------------------------------

generate = st.button(
    "🎨 Generate Image",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# IMAGE GENERATION
# --------------------------------------------------

if generate:

    if not prompt.strip():

        st.error("Please enter a text prompt.")

    else:

        try:

            with st.spinner("Generating your image..."):

                pipe = load_model()

                # Create generator
                generator = torch.Generator(
                    device=device
                ).manual_seed(seed)

                # Generate image
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator
                )

                image = result.images[0]

            # --------------------------------------------------
            # SAVE IMAGE
            # --------------------------------------------------

            os.makedirs("outputs", exist_ok=True)

            output_file = os.path.join(
                "outputs",
                f"generated_{seed}.png"
            )

            image.save(output_file)

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

            st.success("Image generated successfully!")

            st.image(
                image,
                caption=f"Generated Image | Seed: {seed}",
                use_container_width=True
            )

            # --------------------------------------------------
            # DOWNLOAD
            # --------------------------------------------------

            with open(output_file, "rb") as file:

                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name=f"generated_{seed}.png",
                    mime="image/png",
                    use_container_width=True
                )

            # --------------------------------------------------
            # PROMPT INFORMATION
            # --------------------------------------------------

            with st.expander("🔎 Generation Details"):

                st.write("**Prompt:**")
                st.write(prompt)

                st.write("**Negative Prompt:**")
                st.write(negative_prompt)

                st.write("**Width:**", width)
                st.write("**Height:**", height)
                st.write("**Steps:**", steps)
                st.write("**Guidance Scale:**", guidance)
                st.write("**Seed:**", seed)
                st.write("**Device:**", device)

        except Exception as e:

            st.error("Image generation failed.")

            st.exception(e)
