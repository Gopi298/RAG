import os
import re
import tempfile
import streamlit as st
from docx import Document
from gtts import gTTS
from PIL import Image
from pypdf import PdfReader

# MoviePy 2.0+ direct imports
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    ImageClip,
    TextClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

st.set_page_config(page_title="AI Context Video Generator", layout="wide")
st.title("🎥 Long-Form Document & Screenshot Video Generator")
st.write(
    "Upload your PDF, Word document, TXT file, or Screenshot. "
    "The app analyzes the context, splits long content into dynamic scenes, and renders a complete narrated video."
)

# Sidebar settings
st.sidebar.header("Video Configuration")
voice_lang = st.sidebar.selectbox("Voice Language", ["en", "es", "fr", "de"])
bg_color = st.sidebar.color_picker("Background Color", "#0f172a")
text_color = st.sidebar.color_picker("Text Color", "#ffffff")
max_chunk_words = st.sidebar.slider(
    "Words per Scene (Slide)", min_value=20, max_value=100, value=40
)


def extract_text_from_file(uploaded_file):
    """Extract full text from PDF, DOCX, or TXT."""
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


def split_text_into_chunks(text, max_words=40):
    """Split long text into readable scene chunks based on sentences/word count."""
    sentences = re.split(r"(?<=[.!?]) +", text.replace("\n", " "))
    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
        current_chunk.append(sentence)
        current_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


# File Upload Handler
uploaded_file = st.file_uploader(
    "Upload File (PDF, DOCX, TXT, PNG, JPG):",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
)

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # --- MODE 1: LONG DOCUMENT ANALYSIS & MULTI-SCENE VIDEO ---
    if file_type in ["pdf", "docx", "txt"]:
        st.subheader("📄 Document Context Analysis")
        raw_text = extract_text_from_file(uploaded_file)

        if not raw_text:
            st.error("No text could be extracted from this document.")
        else:
            chunks = split_text_into_chunks(raw_text, max_words=max_chunk_words)
            st.success(
                f"Document successfully analyzed! Generated {len(chunks)} scene(s) for the video."
            )

            # Editable preview of chunks
            st.markdown("### Preview Scene Segments:")
            edited_chunks = []
            for i, chunk in enumerate(chunks):
                edited_chunks.append(
                    st.text_area(f"Scene {i+1}", value=chunk, height=80)
                )

            if st.button("🚀 Render Long Document Video"):
                with st.spinner(
                    "Generating multi-scene narration & video rendering..."
                ):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        video_clips = []
                        audio_clips = []

                        for idx, scene_text in enumerate(edited_chunks):
                            if not scene_text.strip():
                                continue

                            # Audio narration for chunk
                            audio_path = os.path.join(
                                temp_dir, f"scene_{idx}.mp3"
                            )
                            tts = gTTS(
                                text=scene_text, lang=voice_lang, slow=False
                            )
                            tts.save(audio_path)

                            a_clip = AudioFileClip(audio_path)
                            scene_duration = a_clip.duration

                            # Visual slide for chunk
                            txt_clip = (
                                TextClip(
                                    font="Arial",
                                    text=scene_text,
                                    font_size=28,
                                    color=text_color,
                                    size=(1280, 720),
                                    method="caption",
                                    bg_color=bg_color,
                                )
                                .with_duration(scene_duration)
                                .with_audio(a_clip)
                            )

                            video_clips.append(txt_clip)
                            audio_clips.append(a_clip)

                        if video_clips:
                            # Concatenate all scene clips into one long video
                            final_video = concatenate_videoclips(
                                video_clips, method="compose"
                            )
                            output_path = os.path.join(
                                temp_dir, "long_document_video.mp4"
                            )

                            final_video.write_videofile(
                                output_path,
                                fps=24,
                                codec="libx264",
                                audio_codec="aac",
                            )

                            # Close clips to prevent memory leaks
                            for c in video_clips:
                                c.close()
                            for a in audio_clips:
                                a.close()

                            st.video(output_path)
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Full Long Video",
                                    data=file,
                                    file_name="full_document_video.mp4",
                                    mime="video/mp4",
                                )

    # --- MODE 2: SCREENSHOT / IMAGE DIALOGUE SCENE ---
    elif file_type in ["png", "jpg", "jpeg"]:
        st.subheader("🖼️ Screenshot / Image Character Dialogue")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        col1, col2 = st.split(2) if hasattr(st, "split") else (st, st)

        char1_text = st.text_area(
            "Male / Character 1 Dialogue:",
            "Based on this document context, here is the first point.",
        )
        char2_text = st.text_area(
            "Female / Character 2 Dialogue:",
            "Got it! Let us review the remaining key insights.",
        )

        if st.button("🚀 Render Character Dialogue Video"):
            with st.spinner("Processing image context and dialogues..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    audio1_p = os.path.join(temp_dir, "dialogue1.mp3")
                    audio2_p = os.path.join(temp_dir, "dialogue2.mp3")
                    img_p = os.path.join(temp_dir, "input_img.png")
                    output_p = os.path.join(temp_dir, "character_dialogue.mp4")

                    image.save(img_p)

                    # 1. Voice generation
                    tts1 = gTTS(text=char1_text, lang=voice_lang)
                    tts1.save(audio1_p)
                    tts2 = gTTS(text=char2_text, lang=voice_lang)
                    tts2.save(audio2_p)

                    aud1 = AudioFileClip(audio1_p)
                    aud2 = AudioFileClip(audio2_p)

                    # 2. Combine sequential speaker audio
                    full_dialogue_audio = concatenate_audioclips([aud1, aud2])

                    # 3. Create video using image background
                    img_clip = (
                        ImageClip(img_p)
                        .resized(height=720)
                        .with_duration(full_dialogue_audio.duration)
                        .with_audio(full_dialogue_audio)
                    )

                    img_clip.write_videofile(
                        output_p, fps=24, codec="libx264", audio_codec="aac"
                    )

                    aud1.close()
                    aud2.close()
                    full_dialogue_audio.close()
                    img_clip.close()

                    st.video(output_p)
                    with open(output_p, "rb") as file:
                        st.download_button(
                            label="📥 Download Character Video",
                            data=file,
                            file_name="character_speech_video.mp4",
                            mime="video/mp4",
                        )
