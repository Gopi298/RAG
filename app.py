import os
import tempfile
import streamlit as st
from docx import Document
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader

# Updated direct imports for MoviePy 2.0+
from moviepy import (
    AudioFileClip,
    ImageClip,
    TextClip,
    concatenate_audioclips,
)

# Streamlit Page Config
st.set_page_config(page_title="AI Video Generator", layout="centered")
st.title("🎥 AI Video Generator")
st.write(
    "Upload a PDF, Word doc, TXT file, or Screenshot/Image to render an MP4 video."
)

# Sidebar options
st.sidebar.header("Video Settings")
voice_lang = st.sidebar.selectbox("Voice Language", ["en", "es", "fr", "de"])
bg_color = st.sidebar.color_picker("Background Color", "#0f172a")
text_color = st.sidebar.color_picker("Text Color", "#ffffff")


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""

    if file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_type == "docx":
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif file_type == "txt":
        text = str(uploaded_file.read(), "utf-8")

    return text.strip()


# File Uploader
uploaded_file = st.file_uploader(
    "Upload your file (PDF, DOCX, TXT, PNG, JPG):",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
)

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # --- SCENARIO 1: DOCUMENT PROCESSING (PDF, DOCX, TXT) ---
    if file_type in ["pdf", "docx", "txt"]:
        st.subheader("📄 Document Processing")
        document_text = extract_text_from_file(uploaded_file)

        if not document_text:
            st.error(
                "Could not extract text from this document. Please try another file."
            )
        else:
            st.success("Text extracted successfully!")
            script = st.text_area(
                "Edit script for voiceover & video text:",
                value=document_text[:1000],
                height=150,
            )

            if st.button("🚀 Generate Document Video"):
                with st.spinner("Generating audio narration and rendering MP4..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_path = os.path.join(temp_dir, "narration.mp3")
                        video_path = os.path.join(temp_dir, "output.mp4")

                        # 1. Generate Voiceover TTS
                        tts = gTTS(text=script, lang=voice_lang, slow=False)
                        tts.save(audio_path)

                        # 2. Get Audio Duration
                        audio_clip = AudioFileClip(audio_path)
                        duration = audio_clip.duration

                        # 3. Create Video Frame with Text (MoviePy 2.0 syntax)
                        txt_clip = (
                            TextClip(
                                font="Arial",
                                text=script,
                                font_size=28,
                                color=text_color,
                                size=(1280, 720),
                                method="caption",
                                bg_color=bg_color,
                            )
                            .with_duration(duration)
                            .with_audio(audio_clip)
                        )

                        # 4. Write Video File
                        txt_clip.write_videofile(
                            video_path, fps=24, codec="libx264", audio_codec="aac"
                        )

                        # Close clips to free memory
                        audio_clip.close()
                        txt_clip.close()

                        # Display Result
                        st.video(video_path)
                        with open(video_path, "rb") as file:
                            st.download_button(
                                label="📥 Download MP4",
                                data=file,
                                file_name="generated_doc_video.mp4",
                                mime="video/mp4",
                            )

    # --- SCENARIO 2: IMAGE/SCREENSHOT PROCESSING ---
    elif file_type in ["png", "jpg", "jpeg"]:
        st.subheader("🖼️ Screenshot / Character Image Processing")
        image = Image.open(uploaded_file)
        st.image(
            image, caption="Uploaded Image", use_container_width=True
        )

        char1_dialogue = st.text_input(
            "Character 1 Speech (e.g., Male Character):",
            "Hello, welcome to our presentation!",
        )
        char2_dialogue = st.text_input(
            "Character 2 Speech (e.g., Female Character):",
            "Thanks! Let us dive into the document summary.",
        )

        if st.button("🚀 Generate Character Speaking Video"):
            with st.spinner("Stitching dialogue audio and processing video..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    audio1_path = os.path.join(temp_dir, "c1.mp3")
                    audio2_path = os.path.join(temp_dir, "c2.mp3")
                    temp_img_path = os.path.join(temp_dir, "input.png")
                    output_video_path = os.path.join(temp_dir, "dialogue.mp4")

                    # Save temporary image
                    image.save(temp_img_path)

                    # Generate voice files for dialogue sequence
                    tts1 = gTTS(text=char1_dialogue, lang=voice_lang)
                    tts1.save(audio1_path)

                    tts2 = gTTS(text=char2_dialogue, lang=voice_lang)
                    tts2.save(audio2_path)

                    clip1 = AudioFileClip(audio1_path)
                    clip2 = AudioFileClip(audio2_path)

                    # Stitch dialogues together sequentially
                    full_audio = concatenate_audioclips([clip1, clip2])
                    total_duration = full_audio.duration

                    # Build Video overlaying dialogue on uploaded image (MoviePy 2.0 syntax)
                    img_clip = (
                        ImageClip(temp_img_path)
                        .resized(height=720)
                        .with_duration(total_duration)
                        .with_audio(full_audio)
                    )

                    img_clip.write_videofile(
                        output_video_path,
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                    )

                    # Close clips
                    clip1.close()
                    clip2.close()
                    full_audio.close()
                    img_clip.close()

                    st.video(output_video_path)
                    with open(output_video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Dialogue Video",
                            data=file,
                            file_name="character_video.mp4",
                            mime="video/mp4",
                        )
