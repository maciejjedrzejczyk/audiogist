import os
import requests
import logging
import json
from app import setup_logging

# Get LM Studio host from environment variable or use default
DEFAULT_LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234")

def get_available_lmstudio_models(host=None):
    """
    Get available models from LM Studio API.
    
    Args:
        host (str): The LM Studio API host (optional, defaults to environment variable or localhost)
        
    Returns:
        list: List of available model names
    """
    logger = setup_logging("lmstudio-models")
    
    if host is None:
        host = DEFAULT_LMSTUDIO_HOST
    
    try:
        response = requests.get(f"{host}/v1/models")
        
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            model_names = [model.get("id") for model in models_data]
            logger.info(f"Found {len(model_names)} LM Studio models: {', '.join(model_names)}")
            return model_names
        else:
            logger.error(f"Error getting LM Studio models: {response.status_code} - {response.text}")
            return []
    
    except Exception as e:
        logger.error(f"Exception when getting LM Studio models: {str(e)}")
        return []

def process_with_lmstudio(prompt, model, host=None):
    """
    Process a prompt with LM Studio API directly.
    
    Args:
        prompt (str): The prompt to process
        model (str): The model to use
        host (str): The LM Studio API host (optional, defaults to environment variable or localhost)
        
    Returns:
        str: The generated response or error message
    """
    logger = setup_logging("lmstudio-api")
    
    if host is None:
        host = DEFAULT_LMSTUDIO_HOST
        
    logger.info(f"Processing prompt with LM Studio model {model}")
    
    # Ensure prompt is a string
    if not isinstance(prompt, str):
        if prompt is None:
            logger.error("Prompt is None, cannot process")
            return "Error: Received None prompt"
        try:
            prompt = str(prompt)
            logger.warning(f"Converted non-string prompt to string, type was: {type(prompt)}")
        except:
            logger.error(f"Failed to convert prompt of type {type(prompt)} to string")
            return "Error: Could not convert prompt to string"
    
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    # Ensure the prompt isn't too long - truncate if necessary
    max_prompt_chars = 12000  # Conservative limit
    if len(prompt) > max_prompt_chars:
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {max_prompt_chars}")
        # Keep beginning and end, cut from middle
        half_size = max_prompt_chars // 2 - 100  # Leave room for [TRUNCATED] marker
        prompt = prompt[:half_size] + "\n\n[...CONTENT TRUNCATED...]\n\n" + prompt[-half_size:]
    
    try:
        response = requests.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that creates concise, coherent summaries."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500  # Limit response length
            },
            timeout=180  # Increased timeout for longer responses
        )
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"LM Studio response received, length: {len(result)}")
            
            # Validate the response - ensure it's not an error message or too short
            if len(result.strip()) < 50 or "error" in result.lower() or "context length" in result.lower():
                logger.warning(f"LM Studio returned potentially invalid response: {result[:100]}...")
                return "Error: LM Studio returned an invalid or incomplete response. Please try again or use a different model."
                
            return result
        else:
            error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error connecting to LM Studio: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

def split_text_into_chunks(text, chunk_size=4000, overlap=200):
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
    
    # Ensure text is a string
    if not isinstance(text, str):
        if text is None:
            return []
        try:
            text = str(text)
        except:
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

def get_efficient_summary_from_lmstudio(transcript_chunks, language, model, custom_prompt=None, 
                                       chunk_size=4000, overlap=200, progress_callback=None):
    """
    Efficiently summarize a long transcript by extracting key points from chunks
    and then generating a single summary from those key points.
    
    Args:
        transcript_chunks (list or str): The transcript chunks or full transcript
        language (str): The language code or name
        model (str): The LM Studio model to use
        custom_prompt (str, optional): Custom prompt template
        chunk_size (int): Size of each chunk in tokens
        overlap (int): Overlap between chunks in tokens
        progress_callback (function, optional): Callback function for progress updates
        
    Returns:
        str: Generated summary or error message
    """
    logger = setup_logging("lmstudio-efficient-summary")
    logger.info(f"Starting efficient LM Studio summarization with model: {model}")
    
    # Handle different input types
    chunks = []
    if isinstance(transcript_chunks, list):
        # If already chunked
        chunks = transcript_chunks
        logger.info(f"Using pre-chunked transcript with {len(chunks)} chunks")
    else:
        # If full transcript, split it
        if not isinstance(transcript_chunks, str):
            if transcript_chunks is None:
                logger.error("Transcript is None, cannot process")
                return "Error: Received None transcript"
            try:
                transcript_text = str(transcript_chunks)
                logger.warning(f"Converted non-string transcript to string, type was: {type(transcript_chunks)}")
            except:
                logger.error(f"Failed to convert transcript of type {type(transcript_chunks)} to string")
                return "Error: Could not convert transcript to string"
        else:
            transcript_text = transcript_chunks
            
        chunks = split_text_into_chunks(transcript_text, chunk_size, overlap)
        logger.info(f"Split transcript into {len(chunks)} chunks")
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        logger.error("No chunks to process")
        return "Error: No content to summarize"
    
    # Language name for prompts
    languages_map = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
        "ru": "Russian", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
        "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "vi": "Vietnamese"
    }
    
    # Handle both language code and full name
    if language in languages_map:
        lang_name = languages_map[language]
    else:
        lang_name = language  # Assume it's already a full name
    
    # STEP 1: Extract key points from each chunk (Map phase)
    if progress_callback:
        progress_callback(0.5, f"Extracting key points from {total_chunks} sections...")
    
    # Simple prompt to extract only the essential points from each chunk
    extract_points_prompt = """
    Extract ONLY the 3-5 most important points from this text section.
    Be extremely concise - use bullet point style.
    Focus only on facts, key information, and main ideas.
    DO NOT include any analysis, summary statements, or introductions.
    
    Text section:
    {chunk}
    
    Key points (in {language}):
    """
    
    all_key_points = []
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(0.5 + (0.3 * (i / total_chunks)), 
                             f"Processing section {i+1}/{total_chunks}...")
        
        # Extract just key points from this chunk
        chunk_text = chunk['chunk_text'] if isinstance(chunk, dict) else chunk
        points = process_with_lmstudio(
            extract_points_prompt.format(chunk=chunk_text, language=lang_name),
            model
        )
        
        # Check if we got a valid response or an error
        if points and not points.startswith("Error:"):
            # Add position context
            position_info = f"From section {i+1}/{total_chunks}:"
            all_key_points.append(f"{position_info}\n{points}")
        else:
            logger.warning(f"Failed to extract key points from chunk {i+1}: {points}")
            # Add a placeholder to maintain section numbering
            all_key_points.append(f"From section {i+1}/{total_chunks}: [Processing error]")
    
    # STEP 2: Generate a comprehensive summary from all key points (Reduce phase)
    if progress_callback:
        progress_callback(0.8, "Generating final summary from all key points...")
    
    # Build the final summary prompt
    if custom_prompt:
        # Adapt custom prompt to work with key points instead of transcript
        summary_prompt = custom_prompt.replace("{transcript}", 
                                              "Key points extracted from the transcript:\n\n" + 
                                              "\n\n".join(all_key_points))
        summary_prompt = summary_prompt.replace("{language}", lang_name)
    else:
        # Default prompt focused on synthesizing from key points
        summary_prompt = f"""
        You are an expert content analyst tasked with creating a COMPREHENSIVE SUMMARY.
        
        Below are key points extracted from different sections of a transcript.
        Create a cohesive, well-structured summary that captures the essential information
        from all these points in {lang_name}.
        
        Key points from the transcript:
        
        {"\n\n".join(all_key_points)}
        
        Your summary should include:
        1. A concise overview (3-5 sentences) that captures the essence of the content
        2. 5-7 key takeaways that represent the most important insights
        3. 1-2 notable quotes or key statements if available
        
        Focus on creating a UNIFIED summary that someone could read INSTEAD OF the full transcript.
        """
    
    # Generate the final summary
    final_summary = process_with_lmstudio(summary_prompt, model)
    
    if progress_callback:
        progress_callback(0.9, "Summary complete")
    
    # Return the final summary - ensure it's a string
    if not final_summary:
        return "Error: Failed to generate summary"
    
    return final_summary

def chat_with_transcript(transcript, question, chat_history=None, model="Llama-3-8B-Instruct", host=None):
    """
    Chat with a transcript using LM Studio.
    
    Args:
        transcript (str): The transcript text
        question (str): The user's question
        chat_history (list, optional): Previous chat history
        model (str): The LM Studio model to use
        host (str, optional): The LM Studio API host
        
    Returns:
        str: The AI-generated response
    """
    logger = setup_logging("lmstudio-chat")
    
    if host is None:
        host = DEFAULT_LMSTUDIO_HOST
    
    # Ensure transcript is a string
    if not isinstance(transcript, str):
        if transcript is None:
            logger.error("Transcript is None, cannot process")
            return "I don't have access to the transcript. Please process a video first."
        try:
            transcript = str(transcript)
        except:
            logger.error(f"Failed to convert transcript of type {type(transcript)} to string")
            return "I'm having trouble reading the transcript. Please try processing the video again."
    
    # Prepare chat history for context
    context = ""
    if chat_history and len(chat_history) > 0:
        # Include up to 3 most recent exchanges for context
        recent_history = chat_history[-min(6, len(chat_history)):]
        context = "Previous conversation:\n"
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        context += "\n"
    
    # Create a prompt that includes the transcript and question
    prompt = f"""You are an AI assistant that answers questions about a transcript.
    
{context}The transcript is as follows:
\"\"\"
{transcript[:15000]}  # Limit transcript length to avoid token limits
\"\"\"

User's question: {question}

Please answer the question based only on information in the transcript. If the answer is not in the transcript, say so clearly. Be concise but thorough."""
    
    try:
        response = requests.post(
            f"{host}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about transcripts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,  # Lower temperature for more factual responses
                "max_tokens": 800  # Limit response length
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"LM Studio chat response received, length: {len(result)}")
            return result
        else:
            error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error connecting to LM Studio: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
