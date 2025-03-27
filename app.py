import streamlit as st
import yt_dlp
import os
import warnings
import whisper
import requests
import tempfile
import glob
import logging
import tiktoken
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("yt_dlp").setLevel(logging.ERROR)

st.set_page_config(
    page_title="AudioGist", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""
if 'current_video_title' not in st.session_state:
    st.session_state.current_video_title = ""

# Supported languages
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
    "Swedish": "sv"
}

# Supported audio file types
SUPPORTED_AUDIO_TYPES = [
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", 
    "audio/ogg", "audio/flac", "audio/aac", "audio/m4a"
]

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to a simple approximation if tiktoken fails
        return len(text.split()) * 1.3  # Rough estimate

# Function to get available Ollama models
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract just the model names from the response
            models = [model['name'] for model in models_data['models']]
            return models
        else:
            st.warning(f"Could not fetch Ollama models: {response.status_code} - {response.text}")
            return ["llama3"]  # Default fallback
    except Exception as e:
        st.warning(f"Error connecting to Ollama: {str(e)}")
        return ["llama3"]  # Default fallback

# Function to get available LM Studio models
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_lmstudio_models():
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            models_data = response.json()
            # Extract just the model names from the response
            models = [model['id'] for model in models_data['data']]
            return models
        else:
            st.warning(f"Could not fetch LM Studio models: {response.status_code} - {response.text}")
            return []  # Default fallback
    except Exception as e:
        st.warning(f"Error connecting to LM Studio: {str(e)}")
        return []  # Default fallback

# Function to download YouTube audio
def download_youtube_audio(url, output_dir, filename_base):
    # Create full path but without extension (yt-dlp will add it)
    output_template = os.path.join(output_dir, filename_base)
    
    # Configure yt-dlp to be quieter
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,  # Reduce console output
        'no_warnings': True,  # Suppress warnings
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Unknown Title')
        
        # Find the actual downloaded file (yt-dlp might add .mp3 extension)
        # Look for any file that starts with our base filename
        possible_files = glob.glob(f"{output_template}*")
        if possible_files:
            # Return the actual file path and the video title
            return possible_files[0], title
        else:
            # Fallback to expected path with .mp3 extension
            return f"{output_template}.mp3", title

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, language, model_size):
    # Suppress warnings during model loading and transcription
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, language=language)
    return result["text"]

# Function to get summary from Ollama
def get_summary_from_ollama(transcript, language, model="llama3", custom_prompt=None, token_limit=4000):
    if custom_prompt:
        prompt = custom_prompt.replace("{transcript}", transcript[:token_limit])
        prompt = prompt.replace("{language}", list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])
    else:
        prompt = f"""
        Please summarize the following transcript and extract key talking points.
        Provide your response in {list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)]}.
        
        Transcript:
        {transcript[:token_limit]}
        
        Please provide:
        1. A concise summary (3-5 sentences)
        2. 5-7 key talking points
        3. Any notable quotes or statements
        """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Function to get response from Ollama for chat
def get_chat_response_from_ollama(transcript, question, language, model="llama3"):
    prompt = f"""
    You are an AI assistant helping with questions about a transcript.
    
    Transcript:
    {transcript[:8000]}  # Using a larger context for questions
    
    User question: {question}
    
    Please provide a helpful, accurate response in {list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)]}.
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Function to get summary from LM Studio
def get_summary_from_lmstudio(transcript, language, model, custom_prompt=None, token_limit=4000):
    if custom_prompt:
        prompt = custom_prompt.replace("{transcript}", transcript[:token_limit])
        prompt = prompt.replace("{language}", list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])
    else:
        prompt = f"""
        Please summarize the following transcript and extract key talking points.
        Provide your response in {list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)]}.
        
        Transcript:
        {transcript[:token_limit]}
        
        Please provide:
        1. A concise summary (3-5 sentences)
        2. 5-7 key talking points
        3. Any notable quotes or statements
        """
    
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that summarizes content."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

# Function to get chat response from LM Studio
def get_chat_response_from_lmstudio(transcript, question, language, model):
    prompt = f"""
    You are an AI assistant helping with questions about a transcript.
    
    Transcript:
    {transcript[:8000]}  # Using a larger context for questions
    
    User question: {question}
    
    Please provide a helpful, accurate response in {list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)]}.
    """
    
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about transcripts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Function to create a clean filename
def clean_filename(title):
    # Create a clean filename from the video title
    clean = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title])
    clean = clean.replace(' ', '_')
    return clean

# Function to process a single YouTube URL
def process_youtube_url(url, language_code, audio_dir, transcript_dir, whisper_model_size, 
                        backend, model, keep_audio, token_limit, custom_prompt):
    results = {
        "success": False,
        "video_title": "",
        "audio_path": "",
        "transcript": "",
        "transcript_path": "",
        "summary": "",
        "summary_path": "",
        "error": ""
    }
    
    try:
        # Download audio
        temp_filename = f"yt_audio_{os.path.basename(tempfile.mktemp())}"
        final_audio_path, video_title = download_youtube_audio(url, audio_dir, temp_filename)
        results["video_title"] = video_title
        
        # Create clean title for output files
        clean_title = clean_filename(video_title)
        
        # Rename the file to use the video title if keep_audio is True
        if keep_audio:
            new_audio_path = os.path.join(audio_dir, f"{clean_title}.mp3")
            if final_audio_path != new_audio_path:
                # Only rename if the paths are different
                if os.path.exists(new_audio_path):
                    # If file already exists, add a timestamp
                    timestamp = int(time.time())
                    new_audio_path = os.path.join(audio_dir, f"{clean_title}_{timestamp}.mp3")
                os.rename(final_audio_path, new_audio_path)
                final_audio_path = new_audio_path
        
        results["audio_path"] = final_audio_path
        
        # Transcribe audio
        transcript = transcribe_audio(final_audio_path, language_code, whisper_model_size)
        results["transcript"] = transcript
        
        # Save transcript
        transcript_filename = f"{clean_title}_transcript.txt"
        transcript_path = os.path.join(transcript_dir, transcript_filename)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        results["transcript_path"] = transcript_path
        
        # Generate summary
        if backend == "Ollama":
            summary = get_summary_from_ollama(transcript, language_code, model, custom_prompt, token_limit)
        else:  # LM Studio
            summary = get_summary_from_lmstudio(transcript, language_code, model, custom_prompt, token_limit)
        
        results["summary"] = summary
        
        # Save summary
        summary_filename = f"{clean_title}_summary.txt"
        summary_path = os.path.join(transcript_dir, summary_filename)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        results["summary_path"] = summary_path
        
        # Clean up if not keeping audio
        if not keep_audio:
            os.remove(final_audio_path)
        
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# Function to save uploaded audio file
def save_uploaded_audio(uploaded_file, output_dir):
    # Create a path for the uploaded file
    file_path = os.path.join(output_dir, uploaded_file.name)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path, uploaded_file.name

# Function to process uploaded audio file
def process_uploaded_audio(uploaded_file, file_title, language_code, audio_dir, transcript_dir, 
                           whisper_model_size, backend, model, keep_audio, token_limit, custom_prompt):
    results = {
        "success": False,
        "video_title": file_title,
        "audio_path": "",
        "transcript": "",
        "transcript_path": "",
        "summary": "",
        "summary_path": "",
        "error": ""
    }
    
    try:
        # Save uploaded file
        audio_path, _ = save_uploaded_audio(uploaded_file, audio_dir)
        results["audio_path"] = audio_path
        
        # Create clean title for output files
        clean_title = clean_filename(file_title)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path, language_code, whisper_model_size)
        results["transcript"] = transcript
        
        # Save transcript
        transcript_filename = f"{clean_title}_transcript.txt"
        transcript_path = os.path.join(transcript_dir, transcript_filename)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        results["transcript_path"] = transcript_path
        
        # Generate summary
        if backend == "Ollama":
            summary = get_summary_from_ollama(transcript, language_code, model, custom_prompt, token_limit)
        else:  # LM Studio
            summary = get_summary_from_lmstudio(transcript, language_code, model, custom_prompt, token_limit)
        
        results["summary"] = summary
        
        # Save summary
        summary_filename = f"{clean_title}_summary.txt"
        summary_path = os.path.join(transcript_dir, summary_filename)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        results["summary_path"] = summary_path
        
        # Clean up if not keeping audio
        if not keep_audio:
            os.remove(audio_path)
        
        results["success"] = True
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# Streamlit UI
st.title("AudioGist")

# Get available Ollama and LM Studio models
available_ollama_models = get_available_ollama_models()
available_lmstudio_models = get_available_lmstudio_models()

# Tabs for different modes
tab1, tab2, tab3 = st.tabs(["Single Video", "Batch Processing", "Chat"])

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Output directory settings
    st.subheader("Output Directories")
    default_download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    
    audio_output_dir = st.text_input(
        "Audio Output Directory", 
        value=default_download_dir,
        help="Directory where audio files will be saved"
    )
    
    transcript_output_dir = st.text_input(
        "Transcript Output Directory", 
        value=default_download_dir,
        help="Directory where transcript files will be saved"
    )
    
    # Whisper model size selection
    st.subheader("Transcription Settings")
    whisper_model_size = st.selectbox(
        "Whisper Model Size", 
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but require more resources"
    )
    
    # Backend selection
    st.subheader("AI Backend Settings")
    backend = st.radio("AI Backend", ["Ollama", "LM Studio"])
    
    # Model selection based on backend
    if backend == "Ollama":
        if not available_ollama_models:
            st.error("No Ollama models found. Please make sure Ollama is running and has models installed.")
            model = "llama3"  # Default fallback
        else:
            model = st.selectbox(
                "Ollama Model", 
                available_ollama_models,
                help="Select the Ollama model to use for summarization"
            )
            
        # Ollama status
        st.subheader("Ollama Status")
        try:
            response = requests.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                version_data = response.json()
                st.success(f"Ollama is running (version: {version_data.get('version', 'unknown')})")
                st.info(f"Available models: {len(available_ollama_models)}")
            else:
                st.error("Ollama is not responding correctly")
        except:
            st.error("Ollama is not running. Please start Ollama service.")
    
    else:  # LM Studio
        if not available_lmstudio_models:
            st.error("No LM Studio models found. Please make sure LM Studio is running and has models loaded.")
            model = ""  # Default fallback
        else:
            model = st.selectbox(
                "LM Studio Model", 
                available_lmstudio_models,
                help="Select the LM Studio model to use for summarization"
            )
            
        # LM Studio status
        st.subheader("LM Studio Status")
        try:
            response = requests.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                st.success("LM Studio is running")
                st.info(f"Available models: {len(available_lmstudio_models)}")
            else:
                st.error("LM Studio is not responding correctly")
        except:
            st.error("LM Studio is not running. Please start LM Studio application.")
    
    # Keep audio file option
    st.subheader("Processing Options")
    keep_audio = st.checkbox("Keep audio file after processing", value=True)
    
    # Custom prompt option
    st.subheader("Customization")
    use_custom_prompt = st.checkbox("Use custom prompt for summarization", value=False)
    
    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Prompt", 
            value="""Please summarize the following transcript and extract key talking points.
Provide your response in {language}.

Transcript:
{transcript}

Please provide:
1. A concise summary (3-5 sentences)
2. 5-7 key talking points
3. Any notable quotes or statements""",
            height=200,
            help="Use {transcript} as placeholder for the transcript and {language} for the language"
        )
    else:
        custom_prompt = None
    
    # Token limit slider
    st.subheader("Token Limit")
    st.info("This limits how much of the transcript is sent to the AI model")
    token_limit = st.slider("Token limit for summarization", 1000, 16000, 4000, 1000)

# Single Video Tab
with tab1:
    st.header("Process Audio")
    
    # Add tabs for YouTube URL and File Upload
    source_tab1, source_tab2 = st.tabs(["YouTube URL", "Upload Audio File"])
    
    with source_tab1:
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        source_type = "youtube"
    
    with source_tab2:
        uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "ogg", "flac", "m4a"])
        file_title = st.text_input("Title (for file naming)", 
                                  placeholder="Enter a title for the audio file...")
        source_type = "upload" if uploaded_file else "youtube"
    
    selected_language = st.selectbox("Select Language", list(LANGUAGES.keys()))
    language_code = LANGUAGES[selected_language]
    
    # Process button
    if st.button("Process Audio", key="process_single"):
        if (source_type == "youtube" and youtube_url) or (source_type == "upload" and uploaded_file):
            try:
                # Check if backend is running
                if backend == "Ollama":
                    try:
                        requests.get("http://localhost:11434/api/version")
                    except:
                        st.error("Ollama is not running. Please start Ollama service before processing.")
                        st.stop()
                else:  # LM Studio
                    try:
                        requests.get("http://localhost:1234/v1/models")
                    except:
                        st.error("LM Studio is not running. Please start LM Studio before processing.")
                        st.stop()
                
                # Ensure directories exist
                audio_dir = ensure_dir(audio_output_dir)
                transcript_dir = ensure_dir(transcript_output_dir)
                
                with st.spinner("Processing..."):
                    # Process the video or audio file
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    if source_type == "youtube":
                        status.info("Downloading YouTube audio...")
                        results = process_youtube_url(
                            youtube_url, 
                            language_code, 
                            audio_dir, 
                            transcript_dir, 
                            whisper_model_size,
                            backend, 
                            model, 
                            keep_audio, 
                            token_limit, 
                            custom_prompt
                        )
                    else:  # upload
                        status.info("Processing uploaded audio...")
                        if not file_title:
                            file_title = os.path.splitext(uploaded_file.name)[0]
                        results = process_uploaded_audio(
                            uploaded_file,
                            file_title,
                            language_code, 
                            audio_dir, 
                            transcript_dir, 
                            whisper_model_size,
                            backend, 
                            model, 
                            keep_audio, 
                            token_limit, 
                            custom_prompt
                        )
                    
                    if results["success"]:
                        # Update progress
                        progress_bar.progress(100)
                        status.success(f"Processed: {results['video_title']}")
                        
                        # Estimate token count
                        transcript = results["transcript"]
                        token_count = estimate_tokens(transcript)
                        
                        # Display transcript
                        st.header("Transcript")
                        st.info(f"Estimated token count: {token_count}")
                        st.text_area("Full Transcript", transcript, height=300)
                        st.info(f"Transcript saved to: {results['transcript_path']}")
                        
                        st.download_button(
                            label="Download Transcript",
                            data=transcript,
                            file_name=f"{clean_filename(results['video_title'])}_transcript.txt",
                            mime="text/plain"
                        )
                        
                        # Display summary
                        st.header("Summary and Key Points")
                        st.text_area("Summary", results["summary"], height=300)
                        st.info(f"Summary saved to: {results['summary_path']}")
                        
                        st.download_button(
                            label="Download Summary",
                            data=results["summary"],
                            file_name=f"{clean_filename(results['video_title'])}_summary.txt",
                            mime="text/plain"
                        )
                        
                        # Store for chat
                        st.session_state.current_transcript = transcript
                        st.session_state.current_video_title = results["video_title"]
                        st.session_state.chat_history = []
                        
                    else:
                        st.error(f"Failed to process: {results['error']}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please either enter a YouTube URL or upload an audio file")

# Batch Processing Tab
with tab2:
    st.header("Process Multiple YouTube Videos")
    
    # Input for multiple URLs
    batch_urls = st.text_area(
        "YouTube URLs (one per line)", 
        height=150,
        placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
    )
    
    batch_language = st.selectbox("Select Language for All Videos", list(LANGUAGES.keys()), key="batch_language")
    batch_language_code = LANGUAGES[batch_language]
    
    max_workers = st.slider("Maximum parallel downloads", 1, 5, 2, 
                           help="Higher values process more videos in parallel but use more resources")
    
    # Process button
    if st.button("Process All Videos", key="process_batch"):
        if batch_urls:
            # Split URLs and filter empty lines
            urls = [url.strip() for url in batch_urls.split("\n") if url.strip()]
            
            if not urls:
                st.warning("Please enter at least one valid YouTube URL")
                st.stop()
            
            # Check if backend is running
            if backend == "Ollama":
                try:
                    requests.get("http://localhost:11434/api/version")
                except:
                    st.error("Ollama is not running. Please start Ollama service before processing.")
                    st.stop()
            else:  # LM Studio
                try:
                    requests.get("http://localhost:1234/v1/models")
                except:
                    st.error("LM Studio is not running. Please start LM Studio before processing.")
                    st.stop()
            
            # Ensure directories exist
            audio_dir = ensure_dir(audio_output_dir)
            transcript_dir = ensure_dir(transcript_output_dir)
            
            # Create a progress container
            progress_container = st.container()
            
            with progress_container:
                st.write(f"Processing {len(urls)} videos...")
                progress_bar = st.progress(0)
                status_area = st.empty()
                
                # Create a results container
                results_container = st.container()
                
                # Process videos in parallel
                all_results = []
                completed = 0
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_url = {
                        executor.submit(
                            process_youtube_url, 
                            url, 
                            batch_language_code, 
                            audio_dir, 
                            transcript_dir, 
                            whisper_model_size,
                            backend, 
                            model, 
                            keep_audio, 
                            token_limit, 
                            custom_prompt
                        ): url for url in urls
                    }
                    
                    # Process as they complete
                    for future in future_to_url:
                        try:
                            result = future.result()
                            all_results.append(result)
                            completed += 1
                            progress_bar.progress(completed / len(urls))
                            
                            if result["success"]:
                                status_area.info(f"Completed {completed}/{len(urls)}: {result['video_title']}")
                            else:
                                status_area.warning(f"Failed {completed}/{len(urls)}: {future_to_url[future]} - {result['error']}")
                        except Exception as e:
                            status_area.error(f"Error processing {future_to_url[future]}: {str(e)}")
                            completed += 1
                            progress_bar.progress(completed / len(urls))
                
                # Final update
                if completed == len(urls):
                    status_area.success(f"Completed processing {len(urls)} videos")
                
                # Display results
                with results_container:
                    st.header("Processing Results")
                    
                    # Count successes and failures
                    successes = sum(1 for r in all_results if r["success"])
                    failures = len(urls) - successes
                    
                    st.write(f"Successfully processed: {successes} videos")
                    if failures > 0:
                        st.write(f"Failed to process: {failures} videos")
                    
                    # Create an expandable section for each video
                    for i, result in enumerate(all_results):
                        if result["success"]:
                            with st.expander(f"Video {i+1}: {result['video_title']}"):
                                st.write(f"**Audio file:** {result['audio_path']}")
                                st.write(f"**Transcript file:** {result['transcript_path']}")
                                st.write(f"**Summary file:** {result['summary_path']}")
                                
                                # Add buttons to view content
                                if st.button(f"View Transcript", key=f"view_transcript_{i}"):
                                    st.text_area("Transcript", result["transcript"], height=200)
                                
                                if st.button(f"View Summary", key=f"view_summary_{i}"):
                                    st.text_area("Summary", result["summary"], height=200)
                        else:
                            with st.expander(f"Video {i+1}: Failed"):
                                st.error(f"Error: {result['error']}")
        else:
            st.warning("Please enter at least one YouTube URL")

# Chat Tab
with tab3:
    st.header("Chat with Transcript")
    
    if not st.session_state.current_transcript:
        st.info("Process a video in the 'Single Video' tab first to enable chat functionality")
    else:
        st.subheader(f"Discussing: {st.session_state.current_video_title}")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        # Input for new question
        user_question = st.chat_input("Ask a question about the transcript...")
        
        if user_question:
            # Display user question
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get response based on backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if backend == "Ollama":
                        response = get_chat_response_from_ollama(
                            st.session_state.current_transcript,
                            user_question,
                            language_code,
                            model
                        )
                    else:  # LM Studio
                        response = get_chat_response_from_lmstudio(
                            st.session_state.current_transcript,
                            user_question,
                            language_code,
                            model
                        )
                    
                    st.write(response)
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, response))

# Add some helpful information at the bottom
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tips")
    st.markdown("""
    - For better transcription quality, use larger Whisper models
    - For faster processing, use smaller models
    - The app works best with clear audio
    - Use batch processing for multiple videos
    - Custom prompts can be used to tailor the summary format
    """)

with col2:
    st.subheader("About")
    st.markdown("""
    This app uses:
    - yt-dlp for YouTube downloads
    - OpenAI's Whisper for transcription
    - Ollama or LM Studio for AI-powered summarization and chat
    
    All processing is done locally on your machine.
    """)

# Add a refresh button for models
if st.button("Refresh Available Models"):
    st.cache_data.clear()
    st.rerun()  # Changed from st.experimental_rerun()