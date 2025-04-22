import os
import requests
import logging
import json
from app import setup_logging

# Get Ollama host from environment variable or use default
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def get_available_ollama_models(host=None):
    """
    Get available models from Ollama API.
    
    Args:
        host (str): The Ollama API host (optional, defaults to environment variable or localhost)
        
    Returns:
        list: List of available model names
    """
    logger = setup_logging("ollama-models")
    
    if host is None:
        host = DEFAULT_OLLAMA_HOST
    
    try:
        response = requests.get(f"{host}/api/tags")
        
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            model_names = [model.get("name") for model in models_data]
            logger.info(f"Found {len(model_names)} Ollama models: {', '.join(model_names)}")
            return model_names
        else:
            logger.error(f"Error getting Ollama models: {response.status_code} - {response.text}")
            return []
    
    except Exception as e:
        logger.error(f"Exception when getting Ollama models: {str(e)}")
        return []

def process_with_ollama(prompt, model, host=None):
    """
    Process a prompt with Ollama API directly.
    
    Args:
        prompt (str): The prompt to process
        model (str): The model to use
        host (str): The Ollama API host (optional, defaults to environment variable or localhost)
        
    Returns:
        str: The generated response or error message
    """
    logger = setup_logging("ollama-api")
    
    if host is None:
        host = DEFAULT_OLLAMA_HOST
        
    logger.info(f"Processing prompt with Ollama model {model}")
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
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            logger.debug(f"Received response from Ollama API: {response_text[:100]}...")
            return response_text
        else:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        logger.error(f"Exception when calling Ollama API: {str(e)}")
        return f"Error: {str(e)}"

def get_efficient_summary_from_ollama(transcript_chunks, language_name, model, custom_prompt=None):
    """
    Generate a summary from transcript chunks using Ollama.
    
    Args:
        transcript_chunks (list): List of transcript chunks
        language_name (str): Language name
        model (str): Ollama model name
        custom_prompt (str, optional): Custom prompt template
        
    Returns:
        str: Generated summary
    """
    logger = setup_logging("ollama-efficient-summary")
    logger.info(f"Starting efficient Ollama summarization with model: {model}")
    
    # Extract key points from each chunk
    all_key_points = []
    
    # Simple prompt to extract only the essential points from each chunk
    extract_points_prompt = f"""
    Extract ONLY the 3-5 most important points from this text section.
    Be extremely concise - use bullet point style.
    Focus only on facts, key information, and main ideas.
    DO NOT include any analysis, summary statements, or introductions.
    
    Text section:
    {{chunk}}
    
    Key points (in {language_name}):
    """
    
    for i, chunk in enumerate(transcript_chunks):
        # Extract just key points from this chunk
        points_prompt = extract_points_prompt.format(chunk=chunk['chunk_text'])
        points = process_with_ollama(points_prompt, model)
        
        if points:
            # Add position context
            position_info = f"From section {i+1}/{len(transcript_chunks)}:"
            all_key_points.append(f"{position_info}\n{points}")
    
    # Generate a comprehensive summary from all key points
    if custom_prompt:
        # Adapt custom prompt to work with key points instead of transcript
        summary_prompt = custom_prompt.replace("{transcript}", 
                                              "Key points extracted from the transcript:\n\n" + 
                                              "\n\n".join(all_key_points))
        summary_prompt = summary_prompt.replace("{language}", language_name)
    else:
        # Default prompt focused on synthesizing from key points
        summary_prompt = f"""
        You are an expert content analyst tasked with creating a COMPREHENSIVE SUMMARY.
        
        Below are key points extracted from different sections of a transcript.
        Create a cohesive, well-structured summary that captures the essential information
        from all these points in {language_name}.
        
        Key points from the transcript:
        
        {"\n\n".join(all_key_points)}
        
        Your summary should include:
        1. A concise overview (3-5 sentences) that captures the essence of the content
        2. 5-7 key takeaways that represent the most important insights
        3. 1-2 notable quotes or key statements if available
        
        Focus on creating a UNIFIED summary that someone could read INSTEAD OF the full transcript.
        """
    
    # Generate the final summary
    final_summary = process_with_ollama(summary_prompt, model)
    
    return final_summary

def chat_with_transcript(transcript, question, chat_history=None, model="llama3", host=None):
    """
    Chat with a transcript using Ollama.
    
    Args:
        transcript (str): The transcript text
        question (str): The user's question
        chat_history (list, optional): Previous chat history
        model (str): The Ollama model to use
        host (str, optional): The Ollama API host
        
    Returns:
        str: The AI-generated response
    """
    logger = setup_logging("ollama-chat")
    
    if host is None:
        host = DEFAULT_OLLAMA_HOST
    
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
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            logger.info(f"Ollama chat response received, length: {len(response_text)}")
            return response_text
        else:
            error_msg = f"Ollama API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Error connecting to Ollama: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
