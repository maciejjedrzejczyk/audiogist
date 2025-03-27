
# AudioGist : Get the gist of audio content without the full listen

**AudioGist** is a powerful application that solves a critical time management problem: determining which long-form audio content deserves your attention without investing hours in listening.

<img src="audiogist.gif" alt="audiogist" width="500" height="800">

AudioGist automatically processes YouTube videos and audio files to generate concise transcripts and AI-powered summaries, enabling users to:

- **Rapidly evaluate content relevance** before committing valuable time to full consumption
- **Extract key insights and talking points** from hours of audio content in minutes
- **Make informed decisions** about which podcasts, interviews, and lectures warrant complete attention
- **Reference and search** previously processed content through a convenient chat interface

## Key Differentiators

1. **Complete Privacy**: All processing occurs locally on the user's machine, ensuring sensitive content never leaves your system
2. **Flexible Input Options**: Process YouTube videos by URL or upload local audio files
3. **Enterprise-Grade AI**: Leverages OpenAI's Whisper for transcription and local LLMs for summarization
4. **Customizable Experience**: Adjustable token limits, custom prompts, and multi-language support

## Features

- **Multiple Input Options**:
  - Download and process YouTube videos by URL
  - Upload and process local audio files
  - Batch process multiple YouTube videos

- **Advanced Transcription**:
  - Transcribe audio using OpenAI's Whisper (runs locally)
  - Support for multiple languages
  - Choose from different model sizes (tiny to large)
  - Automatic token count estimation

- **AI-Powered Summarization**:
  - Generate summaries using local LLMs
  - Support for both Ollama and LM Studio backends
  - Customizable token limits
  - Custom prompt templates

- **Interactive Chat**:
  - Ask questions about the transcript
  - Get AI-generated answers based on the content
  - Persistent chat history

- **File Management**:
  - Save audio files, transcripts, and summaries
  - Customizable output directories
  - Download transcripts and summaries

## Technical Implementation

The application runs on the user's local machine, requiring minimal setup while providing maximum flexibility and privacy. It integrates with popular open-source AI tools ([Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/)) to deliver high-quality summaries without subscription costs or API usage limits.

## Prerequisites

1. **Python 3.7+** with pip

2. **Required Python packages**:
   ```
   streamlit
   yt-dlp
   openai-whisper
   requests
   tiktoken
   ```

3. **Local AI Backend** (at least one of):
   - [Ollama](https://ollama.ai/) - for Linux/macOS
   - [LM Studio](https://lmstudio.ai/) - for Windows/macOS

4. **FFmpeg** (for audio processing):
   - [FFmpeg Installation Guide](https://ffmpeg.org/download.html)

## Installation

1. Clone this repository or download the source code:
   ```bash
   git clone https://github.com/maciejjedrzejczyk/youtube-transcription-tool.git
   cd youtube-transcription-tool
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have either [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/) installed and running locally.

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your web browser.

### Processing a YouTube Video

1. Go to the "Single Video" tab
2. Select the "YouTube URL" tab
3. Paste a YouTube video URL
4. Select the language of the video
5. Click "Process Audio"
6. View and download the transcript and summary

### Processing a Local Audio File

1. Go to the "Single Video" tab
2. Select the "Upload Audio File" tab
3. Upload an audio file from your computer
4. Enter a title for the file (optional)
5. Select the language of the audio
6. Click "Process Audio"
7. View and download the transcript and summary

### Batch Processing

1. Go to the "Batch Processing" tab
2. Enter multiple YouTube URLs (one per line)
3. Select the language for all videos
4. Adjust the number of parallel downloads if needed
5. Click "Process All Videos"
6. View the results for each processed video

### Chatting with Transcripts

1. First process a video or audio file in the "Single Video" tab
2. Go to the "Chat" tab
3. Type your question about the transcript
4. View the AI-generated response
5. Continue the conversation with follow-up questions

## Configuration

### Transcription Settings

- **Whisper Model Size**: Choose between tiny, base, small, medium, or large models
  - Larger models provide better accuracy but require more resources
  - The "base" model is a good balance for most use cases

### AI Backend Settings

- **Backend**: Choose between Ollama and LM Studio
- **Model**: Select from available models in your chosen backend
  - For Ollama, models like llama3, mistral, or gemma work well
  - For LM Studio, any installed model can be used

### Customization

- **Custom Prompt**: Enable to customize the summarization prompt
  - Use {transcript} as a placeholder for the transcript content
  - Use {language} as a placeholder for the selected language

- **Token Limit**: Adjust how much of the transcript is sent to the AI model
  - Higher limits provide more context but may exceed model capabilities
  - 4000 tokens is a good default for most models

## Troubleshooting

- **"Ollama is not running"**: Make sure Ollama is installed and running
- **"LM Studio is not running"**: Make sure LM Studio is running with the API server enabled
- **Transcription errors**: Try using a smaller Whisper model or check your audio quality
- **Missing models**: Use the "Refresh Available Models" button to update the model list