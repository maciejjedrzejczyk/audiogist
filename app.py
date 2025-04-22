import os
import sys
import logging
import streamlit as st
import whisper
import yt_dlp
import requests
import tempfile
import glob
import time
import re
import numpy as np
from datetime import datetime

# Set up logging
SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M")
SESSION_LOG_PATH = None

def setup_logging(task_name="app"):
    """
    Set up logging with a session-based approach.
    
    Args:
        task_name (str): Name of the task/component for the logger
        
    Returns:
        logging.Logger: Configured logger
    """
    global SESSION_LOG_PATH
    
    # Create logger for this task
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.DEBUG)
    
    # If this is the first call, set up the log file
    if SESSION_LOG_PATH is None:
        # Get log directory from environment or use current directory
        log_dir = os.environ.get("AUDIOGIST_LOG_DIR", ".")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with session ID
        SESSION_LOG_PATH = os.path.join(log_dir, f"audiogist-{SESSION_ID}.log")
        
        # Set up file handler for the session
        file_handler = logging.FileHandler(SESSION_LOG_PATH, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger so all loggers use the same file
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Log session start
        logger.info(f"=== New AudioGist session started (ID: {SESSION_ID}) ===")
        logger.info(f"Log file: {SESSION_LOG_PATH}")
    
    return logger

# Set up the main logger
logger = setup_logging()

# Function to download YouTube audio
def download_youtube_audio(url, output_dir="downloads", filename_base=None):
    logger.info(f"download_youtube_audio: url={url}, output_dir={output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output template
    if filename_base:
        output_template = os.path.join(output_dir, filename_base)
    else:
        output_template = os.path.join(output_dir, "%(title)s-%(id)s")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True
    }
    
    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Unknown Title')
    
    # Find the downloaded file
    possible_files = glob.glob(f"{output_template}.mp3")
    if possible_files:
        file_path = possible_files[0]
        logger.info(f"download_youtube_audio: file found {file_path}")
        return file_path, title
    else:
        file_path = f"{output_template}.mp3"
        logger.info(f"download_youtube_audio: fallback to file {file_path}")
        return file_path, title

# Function to download YouTube automatic captions as transcript
def get_youtube_transcript(url, language_code):
    logger.info(f"get_youtube_transcript: url={url}, language_code={language_code}")
    try:
        # Configure yt-dlp to write subtitles
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'skip_download': True,  # Don't download the video
                'writeautomaticsub': True,  # Write automatic subtitles
                'subtitleslangs': [language_code],  # Language preference
                'subtitlesformat': 'vtt',  # Prefer VTT format
                'outtmpl': os.path.join(temp_dir, 'subtitles'),  # Output template
                'quiet': True,
                'no_warnings': True
            }
            
            # Download subtitles
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = info.get('title', 'Unknown Title')
                    
                    # Look for the downloaded subtitle file
                    subtitle_files = glob.glob(os.path.join(temp_dir, '*.vtt'))
                    
                    if not subtitle_files:
                        logger.warning(f"No subtitle files found in {temp_dir}")
                        return None, title
                    
                    # Read and parse the subtitle file
                    with open(subtitle_files[0], 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    # Parse VTT content
                    lines = []
                    in_cue = False
                    
                    for line in vtt_content.splitlines():
                        # Skip headers, timing lines, and empty lines
                        if (line.startswith('WEBVTT') or 
                            '-->' in line or 
                            not line.strip() or
                            line.strip().isdigit()):
                            in_cue = '-->' in line  # Mark start of a cue
                            continue
                        
                        # If we're in a cue block and have text, add it
                        if in_cue and line.strip():
                            lines.append(line.strip())
                    
                    # Join all lines with spaces
                    transcript = ' '.join(lines)
                    
                    logger.info(f"get_youtube_transcript: parsed transcript, length={len(transcript)} chars")
                    logger.debug(f"get_youtube_transcript: transcript snippet={transcript[:200]}")
                    
                    return transcript, title
            except Exception as e:
                logger.error(f"Error in yt-dlp processing: {str(e)}")
                return None, None
    
    except Exception as e:
        logger.error(f"Error getting YouTube transcript: {str(e)}")
        return None, None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, language_code, model_size="base"):
    logger.info(f"transcribe_audio: audio_path={audio_path}, language_code={language_code}, model_size={model_size}")
    try:
        # Load the model
        model = whisper.load_model(model_size)
        
        # Transcribe the audio
        result = model.transcribe(
            audio_path,
            language=language_code,
            verbose=False
        )
        
        # Extract the text
        text = result["text"]
        logger.info(f"Transcription completed, {len(text)} characters transcribed")
        logger.debug(f"Transcript snippet: {text[:200]}")
        return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return f"Error transcribing audio: {str(e)}"

# Function to process a single YouTube video
def process_youtube_video(url, language_code, whisper_model_size, backend, model, token_limit, custom_prompt, chunk_overlap_pct=10, use_youtube_transcript=True):
    try:
        # Validate URL
        if not url or not isinstance(url, str) or not url.strip():
            logger.error("Invalid YouTube URL provided")
            return None, "Invalid YouTube URL provided", None, None, None
            
        # Create a unique filename from the URL
        url_hash = str(hash(url))[-8:]  # Last 8 chars of hash
        filename_base = f"yt_{url_hash}"
        
        with st.status(f"Processing video...") as status:
            # Step 1: Try to get YouTube auto-generated transcript if requested
            transcript = None
            video_title = None
            
            if use_youtube_transcript:
                status.update(label="Fetching YouTube auto-generated captions...")
                try:
                    transcript, video_title = get_youtube_transcript(url, language_code)
                    
                    if transcript:
                        logger.info(f"Using YouTube auto-generated transcript for {url}")
                    else:
                        logger.info(f"No YouTube auto-generated transcript available for {url}, will transcribe audio")
                except Exception as e:
                    logger.error(f"Error fetching YouTube transcript: {str(e)}")
                    transcript = None
            
            # Step 2: If no transcript from YouTube, download and transcribe the audio
            if not transcript:
                try:
                    # Download the audio
                    status.update(label="Downloading audio from YouTube...")
                    audio_path, video_title = download_youtube_audio(url, "downloads", filename_base)
                    
                    # Transcribe the audio
                    status.update(label="Transcribing audio...")
                    transcript = transcribe_audio(audio_path, language_code, whisper_model_size)
                except Exception as e:
                    logger.error(f"Error downloading/transcribing audio: {str(e)}")
                    return None, f"Error: {str(e)}", None, None, None
            
            # Save transcript to file
            transcript_path = os.path.join("transcripts", f"{filename_base}_transcript.txt")
            os.makedirs("transcripts", exist_ok=True)
            
            # Ensure transcript is a string before writing to file
            if isinstance(transcript, list):
                transcript_text = ' '.join(transcript)
            else:
                transcript_text = transcript
                
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            
            # Step 3: Generate summary
            status.update(label="Generating summary...")
            
            # Format the prompt
            language_name = "English"  # Default
            
            # Ensure transcript is a string before formatting prompt
            if isinstance(transcript, list):
                transcript_text = ' '.join(transcript)
            else:
                transcript_text = transcript
                
            formatted_prompt = custom_prompt.format(transcript=transcript_text, language=language_name)
            
            # Check if transcript is too long and needs chunking
            estimated_tokens = estimate_tokens(transcript)
            logger.info(f"Estimated token count for transcript: {estimated_tokens}")
            
            if backend == "Ollama":
                if estimated_tokens > token_limit:
                    # Use RAG approach for Ollama (similar to LM Studio)
                    chunk_overlap = int(token_limit * (chunk_overlap_pct / 100))
                    # Make sure transcript is a string before chunking
                    if isinstance(transcript, list):
                        transcript = ' '.join(transcript)
                    chunks = chunk_text(transcript, token_limit, chunk_overlap)
                    from ollama_client import get_efficient_summary_from_ollama
                    summary = get_efficient_summary_from_ollama(chunks, language_name, model, custom_prompt)
                else:
                    # Make sure transcript is a string
                    if isinstance(transcript, list):
                        transcript = ' '.join(transcript)
                    formatted_prompt = custom_prompt.format(transcript=transcript, language=language_name)
                    from ollama_client import process_with_ollama
                    summary = process_with_ollama(formatted_prompt, model)
            else:  # LM Studio
                try:
                    if estimated_tokens > token_limit:
                        # Use RAG approach for LM Studio
                        chunk_overlap = int(token_limit * (chunk_overlap_pct / 100))
                        # Make sure transcript is a string before chunking
                        if isinstance(transcript, list):
                            transcript = ' '.join(transcript)
                        chunks = chunk_text(transcript, token_limit, chunk_overlap)
                        from lmstudio_client import get_efficient_summary_from_lmstudio
                        summary = get_efficient_summary_from_lmstudio(chunks, language_name, model)
                    else:
                        # Make sure transcript is a string
                        if isinstance(transcript, list):
                            transcript = ' '.join(transcript)
                        formatted_prompt = custom_prompt.format(transcript=transcript, language=language_name)
                        from lmstudio_client import process_with_lmstudio
                        summary = process_with_lmstudio(formatted_prompt, model)
                    
                    # Check if summary is None or error message
                    if not summary:
                        summary = "Error: Failed to generate summary"
                    elif summary.startswith("Error:"):
                        logger.error(f"LM Studio error: {summary}")
                except Exception as e:
                    logger.error(f"Exception in LM Studio processing: {str(e)}")
                    summary = f"Error: {str(e)}"
            
            # Save summary to file
            summary_path = os.path.join("transcripts", f"{filename_base}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            # Store the current transcript and video title in session state for chat
            st.session_state.current_transcript = transcript_text
            st.session_state.current_video_title = video_title
            st.session_state.chat_history = []  # Reset chat history for new video
            
            return transcript_text, summary, video_title, transcript_path, summary_path
    
    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}", exc_info=True)
        return None, f"Error: {str(e)}", None, None, None

# Function to estimate token count
def estimate_tokens(text):
    if not text:
        return 0
    
    # Ensure text is a string
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Simple estimation: ~4 chars per token for English
    return len(text) // 4

# Function to chunk text for RAG processing
def chunk_text(text, chunk_size=4000, overlap=200):
    """
    Split text into chunks with overlap.
    
    Args:
        text (str): The text to split
        chunk_size (int): Approximate size of each chunk in tokens
        overlap (int): Overlap between chunks in tokens
        
    Returns:
        list: List of dictionaries with chunk_text, start_position, and end_position
    """
    if not text:
        return []
    
    # Tokenize the text (simple approximation)
    words = text.split()
    
    # Calculate chunk sizes in words (approximation)
    chunk_words = chunk_size * 3 // 4  # Approximate words per chunk
    overlap_words = overlap * 3 // 4  # Approximate words for overlap
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        # Calculate end index for this chunk
        end_idx = min(start_idx + chunk_words, len(words))
        
        # Extract chunk text
        chunk_text = " ".join(words[start_idx:end_idx])
        
        # Add to chunks list
        chunks.append({
            "chunk_text": chunk_text,
            "start_position": start_idx,
            "end_position": end_idx
        })
        
        # Move start index for next chunk, considering overlap
        start_idx = end_idx - overlap_words if end_idx < len(words) else len(words)
    
    return chunks

# Function to get available Ollama models
def get_available_ollama_models():
    from ollama_client import get_available_ollama_models
    return get_available_ollama_models()

# Function to get available LM Studio models
def get_available_lmstudio_models():
    from lmstudio_client import get_available_lmstudio_models
    return get_available_lmstudio_models()

def main():
    # Set up page config
    st.set_page_config(page_title="AudioGist", page_icon="🎧", layout="wide")
    
    # Set up session state variables if they don't exist
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "youtube"
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    if "lmstudio_models" not in st.session_state:
        st.session_state.lmstudio_models = []
    if "current_transcript" not in st.session_state:
        st.session_state.current_transcript = None
    if "current_video_title" not in st.session_state:
        st.session_state.current_video_title = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Title and description
    st.title("AudioGist")
    st.markdown("Get the gist of audio content without the full listen")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Video", "Batch Processing", "Chat"])
    
    # Tab 1: Single Video Processing
    with tab1:
        st.subheader("Process a Single Video or Audio File")
        
        # Create two columns for input and settings
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Input Options")
            
            # Use radio buttons instead of tabs for clearer state management
            input_mode = st.radio("Input Source", ["YouTube URL", "Upload Audio File"], 
                                 index=0 if st.session_state.input_mode == "youtube" else 1,
                                 horizontal=True)
            
            # Update session state based on selection
            st.session_state.input_mode = "youtube" if input_mode == "YouTube URL" else "upload"
            
            # Show appropriate input based on mode
            if st.session_state.input_mode == "youtube":
                st.session_state.youtube_url = st.text_input("YouTube URL", value=st.session_state.youtube_url)
                youtube_url = st.session_state.youtube_url
                uploaded_file = None
                file_title = None
            else:
                uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "ogg"])
                file_title = st.text_input("Title (optional)")
                youtube_url = None
            
            # Language selection
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Polish", "Russian", "Japanese", "Chinese", "Korean", "Arabic", "Hindi", "Turkish", "Vietnamese"])
        
        with col2:
            # Advanced settings
            with st.expander("Advanced Settings", expanded=True):
                # Backend selection
                backend = st.selectbox("AI Backend", ["Ollama", "LM Studio"])
                
                # Refresh models button
                if st.button("Refresh Available Models"):
                    if backend == "Ollama":
                        st.session_state.ollama_models = get_available_ollama_models()
                    else:  # LM Studio
                        st.session_state.lmstudio_models = get_available_lmstudio_models()
                
                # Get available models based on backend
                if backend == "Ollama":
                    # Use cached models or default list
                    if not st.session_state.ollama_models:
                        try:
                            st.session_state.ollama_models = get_available_ollama_models()
                        except:
                            st.session_state.ollama_models = ["llama3", "mistral", "mixtral"]
                    
                    model = st.selectbox("Model", st.session_state.ollama_models if st.session_state.ollama_models else ["llama3", "mistral", "mixtral"])
                else:  # LM Studio
                    # Use cached models or default list
                    if not st.session_state.lmstudio_models:
                        try:
                            st.session_state.lmstudio_models = get_available_lmstudio_models()
                        except:
                            st.session_state.lmstudio_models = ["Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Mixtral-8x7B-Instruct-v0.1"]
                    
                    model = st.selectbox("Model", st.session_state.lmstudio_models if st.session_state.lmstudio_models else ["Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Mixtral-8x7B-Instruct-v0.1"])
                
                # Whisper model selection
                whisper_model = st.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=1)
                
                # YouTube transcript option
                use_youtube_transcript = st.checkbox("Use YouTube auto-generated captions when available", value=True)
                
                # Custom prompt
                use_custom_prompt = st.checkbox("Use Custom Prompt")
                if use_custom_prompt:
                    custom_prompt = st.text_area("Custom Prompt", "Please summarize the following transcript in {language}:\n\n{transcript}")
                else:
                    custom_prompt = "Please summarize the following transcript in {language}:\n\n{transcript}"
                
                # Token limit
                token_limit = st.slider("Token Limit", 1000, 8000, 4000)
                
                # Chunk overlap (for both backends now)
                chunk_overlap = st.slider("Chunk Overlap Percentage", 0, 50, 10)
        
        # Process button
        if st.button("Process Audio"):
            try:
                # Remove debug info in production
                # st.write(f"Debug - Input mode: {st.session_state.input_mode}")
                # st.write(f"Debug - YouTube URL: '{youtube_url}'")
                
                if st.session_state.input_mode == "youtube" and youtube_url and youtube_url.strip():
                    language_code = language.lower()[:2]  # Simple language code extraction
                    
                    # Log the YouTube URL being processed
                    logger.info(f"Processing YouTube URL: {youtube_url}")
                    
                    transcript, summary, video_title, transcript_path, summary_path = process_youtube_video(
                        youtube_url, language_code, whisper_model, backend, model, token_limit, custom_prompt, chunk_overlap, use_youtube_transcript
                    )
                    
                    if transcript and summary:
                        st.success(f"Successfully processed: {video_title}")
                        
                        # Display transcript and summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Transcript")
                            st.text_area("", transcript, height=400)
                            st.download_button("Download Transcript", transcript, file_name=f"{video_title}_transcript.txt")
                        
                        with col2:
                            st.subheader("Summary")
                            st.text_area("", summary, height=400)
                            st.download_button("Download Summary", summary, file_name=f"{video_title}_summary.txt")
                    else:
                        st.error(f"Failed to process video: {summary if summary else 'Unknown error'}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in Process Audio button handler: {str(e)}", exc_info=True)
                
            # Handle other cases outside the try block
            if uploaded_file:
                st.error("File upload processing not implemented in this simplified example")
            elif not youtube_url or not youtube_url.strip():
                if st.session_state.input_mode == "youtube":
                    st.warning("Please enter a YouTube URL")
                else:
                    st.warning("Please upload an audio file")
    
    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Batch Process YouTube Videos")
        batch_urls = st.text_area("Enter YouTube URLs (one per line)")
        batch_language = st.selectbox("Language for All Videos", ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Polish", "Russian", "Japanese", "Chinese", "Korean", "Arabic", "Hindi", "Turkish", "Vietnamese"])
        parallel_downloads = st.slider("Parallel Downloads", 1, 5, 2)
        
        if st.button("Process All Videos"):
            st.error("Batch processing not implemented in this simplified example")
    
    # Tab 3: Chat with Transcript
    with tab3:
        st.subheader("Chat with Transcript")
        
        if st.session_state.current_transcript:
            st.info(f"Currently loaded: {st.session_state.current_video_title}")
            
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
            
            # Chat input
            user_question = st.chat_input("Ask a question about the transcript")
            if user_question:
                # Add user message to chat history
                st.chat_message("user").write(user_question)
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Process the question with the appropriate backend
                with st.spinner("Thinking..."):
                    try:
                        if backend == "Ollama":
                            from ollama_client import chat_with_transcript as ollama_chat
                            response = ollama_chat(
                                st.session_state.current_transcript,
                                user_question,
                                st.session_state.chat_history[:-1],  # Exclude the current question
                                model
                            )
                        else:  # LM Studio
                            from lmstudio_client import chat_with_transcript as lmstudio_chat
                            response = lmstudio_chat(
                                st.session_state.current_transcript,
                                user_question,
                                st.session_state.chat_history[:-1],  # Exclude the current question
                                model
                            )
                        
                        # Display the response
                        st.chat_message("assistant").write(response)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
                        logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
        else:
            st.warning("Please process a video or audio file first to enable chat.")

if __name__ == "__main__":
    main()
