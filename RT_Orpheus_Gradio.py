# =============================================================================
# Standard Library Imports
# =============================================================================

import json
import logging
import os
import random
import re
import subprocess
import time # Ensure time is imported critical for our logs
import traceback
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple, Union


# =============================================================================
# Third-Party Library Imports
# =============================================================================

import gradio as gr
import numpy as np
import requests
import scipy.io.wavfile as wavfile # Needed for reading audio input path
import torch
from torch import nn


# =============================================================================
# Library Imports with Error Handling & Conditional Imports
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


try:
    import ffmpeg
    logger.info("ffmpeg-python potentially available.")
except ImportError:
    logger.warning("ffmpeg-python not found.")
    ffmpeg = None

try:
    import whisper
    logger.info("Whisper imported.")
except ImportError:
    logger.error("Whisper not found. pip install -U openai-whisper")
    exit(1)

try:
    from snac import SNAC
    logger.info("SNAC imported.")
except ImportError:
    logger.error("SNAC not found. pip install git+https://github.com/hubertsiuzdak/snac.git")
    exit(1)

# =============================================================================
# Configuration Loading
# =============================================================================

# --- API Endpoints & Model Names ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/chat/completions"
TTS_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/completions"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "meta-llama/Llama-3.2-3B-Instruct") # Using Llama 3.2 3B Instruct
TTS_MODEL = os.getenv("TTS_MODEL", "isaiahbjork/orpheus-3b-4bit-quant") # Using Orpheus TTS 3b 4 Quant
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large") # Using Whisper Large model

# --- n8n Configuration ---
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678")
N8N_API_KEY = os.getenv("N8N_API_KEY", "your_n8n_api_key")

# --- Prompts ---
LMSTUDIO_SYSTEM_PROMPT = (
    "Your primary goal is to answer the user's *most recent* question directly and concisely (2-3 short sentences). "
    "The chat history is provided *only* for conversational context. "
    "**Critically Important: Do NOT repeat any sentences or phrases from previous 'assistant' messages in the history.** "
    "Generate a completely fresh response based *only* on the last 'user' message."
    "**Always Remember** Speak in a slow measured tone" 
)

TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_PROMPT_STOP_TOKENS = ["<|eot_id|>", "<|audio|>"]


logger.info(f"Server: {SERVER_BASE_URL}, LLM: {LMSTUDIO_MODEL}, TTS: {TTS_MODEL}, STT: {WHISPER_MODEL_NAME}")


# =============================================================================
# Constants
# =============================================================================

# --- LLM Default Parameters ---
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_LMSTUDIO_MAX_TOKENS = -1  # Use server default
DEFAULT_LMSTUDIO_TEMP = 0.7
DEFAULT_LMSTUDIO_TOP_P = 0.9
DEFAULT_LMSTUDIO_TOP_K = 40
DEFAULT_LMSTUDIO_REP_PENALTY = 1.1
CONTEXT_TURN_LIMIT = 3  # Number of user/assistant pairs in history

# --- TTS Default Parameters ---
DEFAULT_TTS_TEMP = 0.8
DEFAULT_TTS_TOP_P = 0.9
DEFAULT_TTS_TOP_K = 40
DEFAULT_TTS_REP_PENALTY = 1.1

# --- Orpheus/SNAC Specific Constants ---
ORPHEUS_MIN_ID = 10
ORPHEUS_TOKENS_PER_LAYER = 4096
ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)

# --- Audio Processing & Misc ---
TARGET_SAMPLE_RATE = 24000
MAX_SEED = np.iinfo(np.int32).max
TEMP_AUDIO_DIR = "temp_audio_files"

# --- Streaming TTS Constants ---
TTS_STREAM_MIN_GROUPS =40 # Default for UI "Buffer"
TTS_STREAM_SILENCE_MS = 5  # Default for UI "Padding"

# --- Chat History Processing ---
CHAT_HISTORY_ROLES = {"user": "user", "assistant": "assistant"}
CHAT_HISTORY_BOT_EXCLUDE_PREFIXES = (
    "[Error", "(Error", "Sorry,", "(No input", "Processing", "(TTS failed",
    "🔊", "🎧", "🎤", "...", "🧠", "💬 ", "(Empty request)",
    "(TTS generation failed", "(TTS stream generator failed", "(No response)",
    "(No input provided)", "(TTS complete)", "(TTS failed for some sentences)", "(Tara TTS complete)"
)

# --- API Communication ---
API_TIMEOUT_SECONDS = 180
STREAM_TIMEOUT_SECONDS = 300
API_HEADERS = {"Content-Type": "application/json"}
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"

# --- Voice & Tag Constants ---
ALL_VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
TTS_TAGS = {f"@{v}-tts": v for v in ALL_VOICES}
LLM_TAGS = {f"@{v}-llm": v for v in ALL_VOICES}
DEFAULT_TTS_VOICE = ALL_VOICES[0]

# --- UI Message Strings ---
AUDIO_INPUT_PREFIX = "🎤 (Audio Input): "
AUDIO_INPUT_ERROR_PREFIX = f"{AUDIO_INPUT_PREFIX}(Error)"
PLACEHOLDER_MSG = "..."
STREAMING_TTS_MSG = "🔊 Streaming TTS..."
THINKING_MSG = "🧠 Thinking..."
LLM_RESPONSE_PREFIX = "💬 "
EMPTY_REQUEST_MSG = "(Empty request)"
TTS_FAILED_MSG = "(TTS generation failed or produced no audio)"
TTS_INIT_FAILED_MSG = "(TTS stream generator failed to initialize)"
LLM_FAILED_PREFIX = "[Error"
UNEXPECTED_ERROR_MSG_FORMAT = "[An unexpected error occurred: {}]"
NO_INPUT_MSG = "(No input provided)"
NO_RESPONSE_MSG = "(No response)"


# =============================================================================
# Device Setup
# =============================================================================

if torch.cuda.is_available():
    tts_device = "cuda"
    stt_device = "cuda"
else:
    tts_device = "cpu"
    stt_device = "cpu"

logger.info(f"Devices: STT='{stt_device}', TTS='{tts_device}'")


# =============================================================================
# Utility Functions
# =============================================================================

def parse_gguf_codes(response_text: str) -> List[int]:
    """Parse Orpheus <custom_token_ID> from text."""
    try:
        codes = [
            int(m) for m in re.findall(r"<custom_token_(\d+)>", response_text)
            if ORPHEUS_MIN_ID <= int(m) < ORPHEUS_MAX_ID
        ]
        return codes
    except Exception as e:
        logger.error(f"GGUF parse error: {e}")
        return []


def redistribute_codes(codes: List[int], model: nn.Module) -> Optional[np.ndarray]:
    """Convert absolute Orpheus token IDs to SNAC input tensors and decode audio."""
    if not codes or model is None:
        return None

    try:
        dev = next(model.parameters()).device
        layers: List[List[int]] = [[], [], []]
        groups = len(codes) // ORPHEUS_N_LAYERS

        if groups == 0:
            return None

        valid = 0
        for i in range(groups):
            idx = i * ORPHEUS_N_LAYERS
            group = codes[idx : idx + ORPHEUS_N_LAYERS]
            processed: List[Optional[int]] = [None] * ORPHEUS_N_LAYERS
            ok = True

            for j, t_id in enumerate(group):
                if not (ORPHEUS_MIN_ID <= t_id < ORPHEUS_MAX_ID):
                    ok = False; break
                layer_idx = (t_id - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
                code_idx = (t_id - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
                if layer_idx != j:
                    ok = False; break
                processed[j] = code_idx

            if ok:
                try:
                    if any(c is None for c in processed): continue
                    pg: List[int] = processed
                    layers[0].append(pg[0]); layers[1].append(pg[1]); layers[2].append(pg[2])
                    layers[2].append(pg[3]); layers[1].append(pg[4]); layers[2].append(pg[5])
                    layers[2].append(pg[6]); valid += 1
                except (IndexError, TypeError) as map_e:
                    logger.error(f"Code map error in group {i}: {map_e}"); continue

        if valid == 0:
            logger.warning("No valid code groups found after processing.")
            return None
        if not all(layers):
            logger.error("SNAC layers empty after processing valid groups.")
            return None

        tensors = [ torch.tensor(lc, device=dev, dtype=torch.long).unsqueeze(0) for lc in layers ]
        with torch.no_grad():
            audio = model.decode(tensors)

        return audio.detach().squeeze().cpu().numpy()

    except Exception as e:
        logger.exception("SNAC decode error during tensor creation or decoding.")
        return None


def clean_chat_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Clean Gradio chat history (list of dicts) for the LLM API."""
    cleaned_api_format = []
    if not history:
        logger.debug("clean_chat_history received empty history list.")
        return []

    limit = CONTEXT_TURN_LIMIT * 2
    effective_history = history[-limit:]
    logger.debug(f"Cleaning last {len(effective_history)} messages for context (limit: {limit}).")

    if not effective_history:
         logger.warning("Effective history became empty after slicing - unexpected.")
         return []

    for msg in effective_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            logger.warning(f"Skipping invalid message format in history: {msg}")
            continue
        role, content = msg["role"], msg["content"]
        if not content or not isinstance(content, str) or not content.strip():
            continue

        if role == CHAT_HISTORY_ROLES["user"]:
            text = content
            if content.startswith(AUDIO_INPUT_PREFIX):
                text = content.split(AUDIO_INPUT_PREFIX, 1)[1]
            elif content.startswith("@") and " " in content:
                 text = content.split(" ", 1)[1]
            if text and text.strip():
                cleaned_api_format.append({"role": role, "content": text.strip()})
        elif role == CHAT_HISTORY_ROLES["assistant"]:
            text = None
            if not content.startswith(CHAT_HISTORY_BOT_EXCLUDE_PREFIXES) and content != PLACEHOLDER_MSG:
                 text = content[len(LLM_RESPONSE_PREFIX):].strip() if content.startswith(LLM_RESPONSE_PREFIX) else content.strip()
            if text and text.strip():
                cleaned_api_format.append({"role": role, "content": text.strip()})

    return cleaned_api_format


def apply_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: int = 3) -> np.ndarray:
    """Apply a short linear fade-in and fade-out to an audio chunk."""
    num_fade_samples = int(sample_rate * (fade_ms / 1000.0))

    if num_fade_samples <= 0 or audio_chunk.size < 3 * num_fade_samples:
        return audio_chunk

    fade_in = np.linspace(0., 1., num_fade_samples, dtype=audio_chunk.dtype)
    fade_out = np.linspace(1., 0., num_fade_samples, dtype=audio_chunk.dtype)

    chunk_copy = audio_chunk.copy()
    chunk_copy[:num_fade_samples] *= fade_in
    chunk_copy[-num_fade_samples:] *= fade_out

    return chunk_copy


## =============================================================================
# Model Loading
# =============================================================================

logger.info("--- Loading Local Models ---")

logger.info("Loading SNAC vocoder model...")
snac_model: Optional[SNAC] = None
try: # Load SNAC
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="snac.snac")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

    if snac_model:
        snac_model = snac_model.to(tts_device).eval()
        logger.info(f"SNAC loaded to '{tts_device}'.")
    else:
        logger.error("SNAC.from_pretrained returned None. Model not loaded.")
        snac_model = None

except Exception as e:
    logger.exception("Fatal error loading SNAC.")
    snac_model = None

if not snac_model:
    logger.critical("SNAC model failed to load. TTS disabled.")
else:
    try:
        logger.info("Attempting SNAC warm-up...")
        dummy_tokens = [
            min(ORPHEUS_MIN_ID + i * ORPHEUS_TOKENS_PER_LAYER + 100, ORPHEUS_MAX_ID - 1)
            for i in range(ORPHEUS_N_LAYERS)
        ]
        warmup_audio = redistribute_codes(dummy_tokens, snac_model)
        if warmup_audio is not None:
            logger.info("SNAC warm-up OK.")
        else:
             logger.warning("SNAC warm-up call ran but produced no audio.")
    except Exception as wu_e:
        logger.exception("SNAC warm-up call failed with an exception.")


logger.info(f"Loading Whisper Large model ({WHISPER_MODEL_NAME})...")
whisper_model: Optional[whisper.Whisper] = None
try: # Load Whisper
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    if stt_device == "cpu":
        warnings.filterwarnings("ignore", message=".*FP16 is not supported.*")
    
    # For large model, we need more memory
    if stt_device == "cuda":
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before loading Whisper Large")
    
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=stt_device)
    logger.info(f"Whisper '{WHISPER_MODEL_NAME}' loaded successfully to '{stt_device}'.")
    
    # Warm up Whisper model
    logger.info("Warming up Whisper model...")
    dummy_audio = np.zeros((TARGET_SAMPLE_RATE,), dtype=np.float32)
    whisper_model.transcribe(dummy_audio, language="en")
    logger.info("Whisper warm-up complete.")

except Exception as e:
    logger.exception(f"Fatal error loading Whisper: {str(e)}")
    whisper_model = None

if not whisper_model:
    logger.critical("Whisper model failed to load. Audio input disabled.")


logger.info("--- Local Model Loading Complete ---")


# =============================================================================
# TTS Pipeline Function (Streaming for Direct TTS Mode)
# =============================================================================

def generate_speech_stream(
    text: str,
    voice: str,
    tts_temperature: float,
    tts_top_p: float,
    tts_repetition_penalty: float,
    buffer_groups_param: int,
    padding_ms_param: int,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Generates audio chunks via TTS streaming API + local SNAC, applies fade and padding."""

    # Pre-checks
    if not text.strip():
        logger.warning("generate_speech_stream called with empty text.")
        return
    if snac_model is None:
        logger.error("generate_speech_stream called but snac_model is not loaded.")
        return

    # Optimize buffer size for 4-bit quantized model
    min_codes_required = max(buffer_groups_param * ORPHEUS_N_LAYERS, 28)  # Ensure minimum buffer size for stable output
    logger.debug(f"Stream processing: buffer={buffer_groups_param} groups ({min_codes_required} codes), padding={padding_ms_param} ms")

    silence_samples: int = 0
    if padding_ms_param > 0:
        silence_samples = int(TARGET_SAMPLE_RATE * (padding_ms_param / 1000.0))
        logger.debug(f"Calculated silence samples per side: {silence_samples}")

    # Optimized payload for 4-bit quantized model
    payload = {
        "model": TTS_MODEL,
        "prompt": TTS_PROMPT_FORMAT.format(voice=voice, text=text),
        "temperature": max(0.1, min(tts_temperature, 0.9)),  # Clamp temperature for stability
        "top_p": max(0.1, min(tts_top_p, 0.95)),  # Clamp top_p for stability
        "repeat_penalty": max(1.0, min(tts_repetition_penalty, 1.3)),  # Clamp repeat penalty
        "n_predict": -1,
        "stop": TTS_PROMPT_STOP_TOKENS,
        "stream": True,
        "mirostat": 2,  # Enable Mirostat 2.0 sampling for better stability
        "mirostat_tau": 5.0,  # Target entropy (lower = more focused)
        "mirostat_eta": 0.1,  # Learning rate
    }

    accumulated_codes: List[int] = []
    response = None
    stream_start_time = time.time()
    last_chunk_time = time.time()
    chunk_timeout = 10.0  # Timeout for receiving chunks

    try:
        logger.info(">>> TTS API: Initiating stream request...")
        with requests.post(
            TTS_API_ENDPOINT, 
            json=payload, 
            headers=STREAM_HEADERS, 
            stream=True, 
            timeout=STREAM_TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            logger.info(f"--- TTS API: Stream connected after {time.time() - stream_start_time:.3f}s. Receiving codes...")

            for line in response.iter_lines():
                current_time = time.time()
                if current_time - last_chunk_time > chunk_timeout:
                    logger.warning("TTS stream chunk timeout exceeded")
                    break
                
                if not line: continue
                try: 
                    decoded_line = line.decode(response.encoding or 'utf-8')
                except UnicodeDecodeError: 
                    logger.warning(f"Skipping undecodable line: {line[:50]}...")
                    continue

                if decoded_line.startswith(SSE_DATA_PREFIX):
                    json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                    if json_str == SSE_DONE_MARKER: 
                        logger.debug("Received TTS SSE_DONE_MARKER.")
                        break
                    if not json_str: continue

                    try:
                        data = json.loads(json_str)
                        chunk_text = ""
                        if "content" in data: 
                            chunk_text = data.get("content", "")
                        elif "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            chunk_text = delta.get("content", "") or choice.get("text", "")

                        if chunk_text:
                            last_chunk_time = current_time
                            new_codes = parse_gguf_codes(chunk_text)
                            if new_codes:
                                accumulated_codes.extend(new_codes)
                                if len(accumulated_codes) >= min_codes_required:
                                    num_groups_to_decode = len(accumulated_codes) // ORPHEUS_N_LAYERS
                                    codes_to_decode = accumulated_codes[:num_groups_to_decode * ORPHEUS_N_LAYERS]
                                    accumulated_codes = accumulated_codes[num_groups_to_decode * ORPHEUS_N_LAYERS:]

                                    snac_start_time = time.time()
                                    audio_chunk = redistribute_codes(codes_to_decode, snac_model)
                                    snac_end_time = time.time()

                                    if audio_chunk is not None and audio_chunk.size > 0:
                                        logger.debug(f"--- SNAC: Decoded chunk ({len(codes_to_decode)} codes -> {audio_chunk.size} samples) in {snac_end_time - snac_start_time:.3f}s.")
                                        # Apply longer fade for smoother transitions
                                        faded_chunk = apply_fade(audio_chunk, TARGET_SAMPLE_RATE, fade_ms=5)

                                        if silence_samples > 0:
                                            silence = np.zeros(silence_samples, dtype=faded_chunk.dtype)
                                            yield (TARGET_SAMPLE_RATE, np.concatenate((silence, faded_chunk, silence)))
                                        else:
                                            yield (TARGET_SAMPLE_RATE, faded_chunk)
                                    else:
                                        logger.warning(f"--- SNAC: Failed to decode chunk ({len(codes_to_decode)} codes) in {snac_end_time - snac_start_time:.3f}s.")

                        stop_reason = None
                        is_stopped = False
                        if "choices" in data and data["choices"]:
                            stop_reason = data["choices"][0].get("finish_reason")
                        if stop_reason or data.get("stop") or data.get("stopped_eos") or data.get("stopped_limit"):
                            is_stopped = True
                            logger.debug(f"TTS Stream stop condition met: reason='{stop_reason}', data flags: {data.get('stop')}, {data.get('stopped_eos')}, {data.get('stopped_limit')}")
                        if is_stopped: 
                            break

                    except json.JSONDecodeError: 
                        logger.warning(f"Skipping invalid JSON in TTS stream: {json_str[:100]}...")
                        continue
                    except Exception as e: 
                        logger.exception(f"Error processing TTS stream chunk: {json_str[:100]}...")
                        continue

            # Process remaining codes
            if len(accumulated_codes) >= ORPHEUS_N_LAYERS:
                logger.debug(f"Processing final {len(accumulated_codes)} codes after stream end.")
                num_groups = len(accumulated_codes) // ORPHEUS_N_LAYERS
                codes_to_decode = accumulated_codes[:num_groups * ORPHEUS_N_LAYERS]

                snac_start_time = time.time()
                audio_chunk = redistribute_codes(codes_to_decode, snac_model)
                snac_end_time = time.time()

                if audio_chunk is not None and audio_chunk.size > 0:
                    logger.debug(f"--- SNAC: Decoded final chunk ({len(codes_to_decode)} codes -> {audio_chunk.size} samples) in {snac_end_time - snac_start_time:.3f}s.")
                    faded_chunk = apply_fade(audio_chunk, TARGET_SAMPLE_RATE, fade_ms=5)
                    if silence_samples > 0:
                        silence = np.zeros(silence_samples, dtype=faded_chunk.dtype)
                        yield (TARGET_SAMPLE_RATE, np.concatenate((silence, faded_chunk, silence)))
                    else:
                        yield (TARGET_SAMPLE_RATE, faded_chunk)
                else:
                    logger.warning(f"--- SNAC: Failed to decode final chunk ({len(codes_to_decode)} codes) in {snac_end_time - snac_start_time:.3f}s.")
            else:
                logger.debug(f"Discarding final {len(accumulated_codes)} codes (less than {ORPHEUS_N_LAYERS}).")

    except requests.exceptions.RequestException as e:
        logger.exception(f"<<< TTS API: RequestException after {time.time() - stream_start_time:.3f}s.")
    except Exception as e:
        logger.exception(f"<<< TTS API: Unexpected error after {time.time() - stream_start_time:.3f}s.")
    finally:
        logger.info(f"<<< TTS API: Stream processing finished after {time.time() - stream_start_time:.3f}s (relative to request start).")


# =============================================================================
# LLM Communication Function (Streaming)
# =============================================================================

def call_lmstudio_streaming(
    lmstudio_payload: Dict[str, Any],
    generation_params: Dict[str, Any]
) -> Generator[str, None, None]:
    """Calls the LLM API for a STREAMING chat response."""
    try:
        # Optimized parameters for Llama 3.2 3B
        payload = {
            "model": LMSTUDIO_MODEL,
            "messages": lmstudio_payload.get("messages", []),
            "temperature": generation_params.get('lmstudio_temperature', DEFAULT_LMSTUDIO_TEMP),
            "top_p": generation_params.get('lmstudio_top_p', DEFAULT_LMSTUDIO_TOP_P),
            "max_tokens": generation_params.get('lmstudio_max_new_tokens', DEFAULT_LMSTUDIO_MAX_TOKENS),
            "repeat_penalty": generation_params.get('lmstudio_repetition_penalty', DEFAULT_LMSTUDIO_REP_PENALTY),
            "top_k": generation_params.get('lmstudio_top_k', DEFAULT_LMSTUDIO_TOP_K),
            "stream": True,
            # Additional parameters optimized for Llama
            "presence_penalty": 0.0,  # Helps with repetition
            "frequency_penalty": 0.0,  # Helps with repetition
            "stop": ["</s>", "[/INST]"],  # Llama specific stop tokens
        }
        
        if payload.get("max_tokens") == -1: 
            payload["max_tokens"] = None
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.debug(f"Initiating LLM stream request to {LMSTUDIO_API_ENDPOINT} with model {LMSTUDIO_MODEL}")
        with requests.post(
            LMSTUDIO_API_ENDPOINT, 
            json=payload, 
            headers=STREAM_HEADERS, 
            stream=True, 
            timeout=STREAM_TIMEOUT_SECONDS
        ) as response:
            response.raise_for_status()
            error_occurred = False
            accumulated_text = ""
            
            for line in response.iter_lines():
                if error_occurred: break
                if not line: continue
                
                try: 
                    decoded_line = line.decode("utf-8")
                except UnicodeDecodeError: 
                    logger.warning(f"Skipping undecodable line in LLM stream: {line[:50]}...") 
                    continue

                if decoded_line.startswith(SSE_DATA_PREFIX):
                    json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                    if json_str == SSE_DONE_MARKER: 
                        logger.debug("Received LLM SSE_DONE_MARKER.")
                        break
                    
                    if not json_str: continue
                    
                    try:
                        data = json.loads(json_str)
                        delta_content = None
                        
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            delta = choice.get("delta",{})
                            delta_content = delta.get("content")
                            
                        if delta_content is not None:
                            # Clean up Llama specific artifacts
                            delta_content = delta_content.replace("</s>", "").replace("[/INST]", "")
                            accumulated_text += delta_content
                            
                            # Only yield if we have a meaningful chunk of text
                            if len(accumulated_text) >= 4 or "." in accumulated_text or "?" in accumulated_text or "!" in accumulated_text:
                                yield accumulated_text
                                accumulated_text = ""
                            
                        elif "error" in data:
                            error_msg = data.get('error', {}).get('message', 'Unknown error')
                            formatted_error = f"[Error from LLM Stream: {error_msg}]"
                            logger.error(formatted_error)
                            yield formatted_error
                            error_occurred = True
                            break
                            
                    except json.JSONDecodeError: 
                        logger.warning(f"Skipping invalid JSON in LLM stream: {json_str[:100]}...")
                        continue
                    except Exception as e: 
                        logger.exception(f"Error processing LLM stream chunk: {json_str[:100]}...")
                        err_yield = f"[Error processing LLM stream: {e}]"
                        yield err_yield
                        error_occurred = True
                        break
            
            # Yield any remaining accumulated text
            if accumulated_text:
                yield accumulated_text
                
    except requests.exceptions.RequestException as e:
        logger.exception("RequestException occurred during LLM stream.")
        err_yield = "[Error connecting to LLM]"
        if hasattr(e, 'response') and e.response is not None:
             logger.error(f"LLM Stream Error Status: {e.response.status_code}, Body: {e.response.text[:500]}")
             try:
                 err_json=e.response.json()
                 detail = err_json.get('error',{}).get('message') or err_json.get('detail')
                 if detail: 
                     err_yield = f"[Error from LLM Server: {detail}]"
             except: 
                 pass
        yield err_yield
    except Exception as e: 
        logger.exception("Unexpected error in call_lmstudio_streaming.")
        yield f"[Unexpected Error during LLM stream: {e}]"


# =============================================================================
# Gradio UI Helper Function
# =============================================================================

def _determine_processing_details(
    input_text: str,
    current_mode: str,
    tts_tags: Dict[str, str],
    llm_tags: Dict[str, str]
) -> Tuple[str, str, str, Optional[str]]:
    """Determines effective mode, voice, and text based on input/tags."""
    input_text = input_text.strip() if input_text else ""; lower_text = input_text.lower()
    effective_mode, chosen_voice, text_for_processing, command_tag_used = current_mode, DEFAULT_TTS_VOICE, input_text, None
    all_tags = {**tts_tags, **llm_tags}
    for tag in all_tags.keys():
        if lower_text == tag or lower_text.startswith(tag + " "):
            command_tag_used = tag; chosen_voice = all_tags[tag]; text_for_processing = input_text[len(tag) :].strip()
            effective_mode = "tts" if tag in tts_tags else "llm"; logger.info(f"Tag '{tag}' detected, overriding UI mode. Effective Mode='{effective_mode}', Voice='{chosen_voice}'.")
            return text_for_processing, effective_mode, chosen_voice, command_tag_used
    logger.debug(f"No command tag found. Using UI mode '{current_mode}'.")
    return text_for_processing, effective_mode, chosen_voice, command_tag_used


# =============================================================================
# Main Gradio Processing Function (Generator)
# =============================================================================

# *** UPDATED SIGNATURE to include log_state ***
def process_input_blocks(
    # Inputs from Gradio components
    text_input: Optional[str],
    audio_input_path: Optional[str],
    mode: str,

    # LLM parameters
    lmstudio_max_new_tokens: int,
    lmstudio_temperature: float,
    lmstudio_top_p: float,
    lmstudio_top_k: int,
    lmstudio_repetition_penalty: float,

    # TTS API parameters
    tts_temperature: float,
    tts_top_p: float,
    tts_repetition_penalty: float,

    # Buffer and Padding parameters from UI
    tts_buffer_groups: int,
    tts_padding_ms: int,

    # Chat state
    chat_history_value: List[Dict[str, str]],
    # *** NEW: Log state ***
    log_state_value: str,

) -> Generator[Dict[gr.component, Any], None, None]:
    """Main processing function using chunked streaming TTS with fades and padding."""
    # *** Make log components accessible (assuming global scope within Gradio) ***
    global chatbot_state, chatbot_display, text_input_box, audio_input_mic, streaming_audio, log_output, log_state

    # --- Utility function to update UI log ---
    def _update_log(message: str, current_log: str) -> str:
        timestamp = time.strftime("%H:%M:%S")
        new_log_line = f"{timestamp} - {message}"
        # Append new message, keeping the log from getting excessively long in the state
        log_lines = current_log.splitlines()
        max_log_lines_in_state = 50 # Keep roughly this many lines in the background state
        if len(log_lines) > max_log_lines_in_state:
            log_lines = log_lines[-max_log_lines_in_state:]
        updated_log = "\n".join(log_lines) + "\n" + new_log_line
        return updated_log.strip() # Return stripped log
    # --- End utility ---

    process_start_time = time.time()
    current_log = "" # Start with empty log for this run
    start_message = f"--- Processing Input --- UI Mode: {mode} ---"
    logger.info(start_message)
    current_log = _update_log(start_message, current_log)
    yield { log_state: current_log, log_output: current_log } # Show start message

    current_chat_history = list(chat_history_value) if chat_history_value else []
    user_display_input, user_message_content, audio_filepath_to_clean = None, None, None

    # --- 1. Process Audio Input ---
    stt_duration = 0.0
    if audio_input_path and whisper_model and os.path.isfile(audio_input_path):
        stt_start_time = time.time()
        stt_start_msg = f">>> STT: Starting transcription..."
        logger.info(stt_start_msg)
        current_log = _update_log(stt_start_msg, current_log)
        yield { log_state: current_log, log_output: current_log }
        try:
            result = whisper_model.transcribe(audio_input_path, fp16=(stt_device == "cuda"))
            original_text = result["text"].strip()
            stt_end_time = time.time()
            stt_duration = stt_end_time - stt_start_time
            stt_end_msg = f"<<< STT: Finished in {stt_duration:.3f}s. Result: '{original_text[:30]}...'"
            logger.info(stt_end_msg)
            current_log = _update_log(stt_end_msg, current_log)
            yield { log_state: current_log, log_output: current_log }

            user_display_input = f"{AUDIO_INPUT_PREFIX}{original_text}"
            user_message_content = original_text
            audio_filepath_to_clean = audio_input_path # Still track it even if not deleted
        except Exception as e:
            stt_end_time = time.time()
            stt_duration = stt_end_time - stt_start_time
            error_msg = f"[Error during transcription: {e}]"
            stt_fail_msg = f"<<< STT: FAILED after {stt_duration:.3f}s."
            logger.exception(stt_fail_msg)
            current_log = _update_log(stt_fail_msg + f" Error: {e}", current_log)
            current_chat_history.extend([{"role": "user", "content": AUDIO_INPUT_ERROR_PREFIX}, {"role": "assistant", "content": error_msg}])
            yield { chatbot_state: current_chat_history, chatbot_display: current_chat_history, text_input_box: "", audio_input_mic: None, streaming_audio: None, log_state: current_log, log_output: current_log }; return

    # --- 2. Process Text Input ---
    if not user_message_content and text_input:
        text_input_msg = "--- Input: Using text input."
        logger.info(text_input_msg)
        # current_log = _update_log(text_input_msg, current_log) # Too verbose for UI
        # yield { log_state: current_log, log_output: current_log }
        original_text = text_input.strip()
        user_display_input = original_text
        user_message_content = original_text

    # --- 3. Handle No Input ---
    if not user_message_content or not user_message_content.strip():
        no_input_msg = "--- Input: No valid input provided.";
        logger.warning(no_input_msg)
        current_log = _update_log(no_input_msg, current_log)
        current_chat_history.extend([{"role": "user", "content": NO_INPUT_MSG}, {"role": "assistant", "content": NO_RESPONSE_MSG}])
        yield { chatbot_state: current_chat_history, chatbot_display: current_chat_history, text_input_box: "", audio_input_mic: None, streaming_audio: None, log_state: current_log, log_output: current_log };
        return

    # --- 4. Determine Effective Processing Details ---
    text_to_process, effective_mode, chosen_voice, command_tag_used = _determine_processing_details(user_message_content, mode, TTS_TAGS, LLM_TAGS)
    config_msg = f"--- Config: Effective Mode='{effective_mode}', Voice='{chosen_voice}'"
    logger.info(config_msg + f", Text='{text_to_process[:50]}...'") # Add text to console log
    current_log = _update_log(config_msg, current_log) # UI log doesn't need full text
    yield { log_state: current_log, log_output: current_log }


    # --- 5. Initial UI Update ---
    current_chat_history.extend([{"role": "user", "content": user_display_input}, {"role": "assistant", "content": PLACEHOLDER_MSG}])
    yield { chatbot_state: current_chat_history, chatbot_display: current_chat_history, text_input_box: "", audio_input_mic: None, log_state: current_log, log_output: current_log }

    # --- 6. Core Processing Logic ---
    final_bot_message_text_clean = ""
    final_yield_dict = {}
    pipeline_start_time = time.time() # Start timing the core LLM/TTS work

    try:
        # --- Branch A: Direct TTS ---
        if effective_mode == "tts":
            tts_mode_msg = ">>> TTS Pipeline: Starting Direct TTS Mode..."
            logger.info(tts_mode_msg)
            current_log = _update_log(tts_mode_msg, current_log)
            yield { log_state: current_log, log_output: current_log }

            if not text_to_process.strip(): final_bot_message_text_clean = EMPTY_REQUEST_MSG
            elif not snac_model: logger.error("--- Error: Direct TTS requested but SNAC unavailable."); final_bot_message_text_clean = "[Error: SNAC unavailable]"
            else:
                streaming_disp = current_chat_history[:-1] + [{"role": "assistant", "content": STREAMING_TTS_MSG}]
                yield { chatbot_display: streaming_disp, log_state: current_log, log_output: current_log }

                tts_pipeline_start_time = time.time()
                audio_gen = generate_speech_stream(
                    text_to_process, chosen_voice, tts_temperature, tts_top_p, tts_repetition_penalty,
                    buffer_groups_param=tts_buffer_groups, padding_ms_param=tts_padding_ms
                )
                stream_ok, first_yield = False, False
                first_audio_yield_time = None

                for sr, audio_chunk in audio_gen:
                     if not first_yield and audio_chunk is not None and audio_chunk.size > 0 :
                         first_audio_yield_time = time.time()
                         ttfa_msg = f"--- TTS Pipeline: First audio chunk ready after {first_audio_yield_time - tts_pipeline_start_time:.3f}s."
                         logger.info(ttfa_msg)
                         current_log = _update_log(ttfa_msg, current_log)
                         yield { streaming_audio: (sr, audio_chunk), log_state: current_log, log_output: current_log }
                         stream_ok = True
                     elif audio_chunk is not None and audio_chunk.size > 0:
                         yield { streaming_audio: (sr, audio_chunk) }
                         stream_ok = True
                     first_yield = True

                tts_pipeline_end_time = time.time()
                tts_end_msg = f"<<< TTS Pipeline: Direct TTS finished in {tts_pipeline_end_time - tts_pipeline_start_time:.3f}s."
                logger.info(tts_end_msg)
                current_log = _update_log(tts_end_msg, current_log)
                yield { log_state: current_log, log_output: current_log }


                if stream_ok: final_bot_message_text_clean = f"({chosen_voice.capitalize()} TTS complete)"
                elif audio_gen is not None: final_bot_message_text_clean = TTS_FAILED_MSG
                else: final_bot_message_text_clean = TTS_INIT_FAILED_MSG
                if not first_yield: final_yield_dict[streaming_audio] = None

        # --- Branch B: LLM + TTS ---
        elif effective_mode == "llm":
            llm_mode_msg = ">>> LLM+TTS Pipeline: Starting LLM Mode..."
            logger.info(llm_mode_msg)
            current_log = _update_log(llm_mode_msg, current_log)
            yield { log_state: current_log, log_output: current_log }

            if not text_to_process.strip(): final_bot_message_text_clean = EMPTY_REQUEST_MSG
            elif not snac_model: logger.error("--- Error: LLM+TTS requested but SNAC unavailable."); final_bot_message_text_clean = "[Error: SNAC unavailable]"
            else:
                llm_history = clean_chat_history(current_chat_history[:-2])
                llm_messages = [{"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}] + llm_history + [{"role": "user", "content": text_to_process}]
                llm_payload = {"messages": llm_messages}
                gen_params = {"lmstudio_temperature": lmstudio_temperature, "lmstudio_top_p": lmstudio_top_p, "lmstudio_max_new_tokens": lmstudio_max_new_tokens, "lmstudio_repetition_penalty": lmstudio_repetition_penalty, "lmstudio_top_k": lmstudio_top_k }

                llm_text_buffer = ""; llm_error = False; first_llm_chunk_processed = False
                llm_stream_start_time = time.time()
                llm_start_msg = ">>> LLM API: Initiating stream..."
                logger.info(llm_start_msg)
                current_log = _update_log(llm_start_msg, current_log)
                yield { log_state: current_log, log_output: current_log }
                llm_gen = call_lmstudio_streaming(llm_payload, gen_params)

                for chunk in llm_gen:
                    if not first_llm_chunk_processed and chunk is not None:
                        first_token_time = time.time()
                        llm_first_token_msg = f"--- LLM API: First token received after {first_token_time - llm_stream_start_time:.3f}s."
                        logger.info(llm_first_token_msg)
                        current_log = _update_log(llm_first_token_msg, current_log)
                        first_llm_chunk_processed = True

                    if chunk.startswith(LLM_FAILED_PREFIX):
                         final_bot_message_text_clean = chunk; llm_error = True
                         llm_stream_end_time = time.time()
                         llm_fail_msg = f"<<< LLM API: Stream ended with error after {llm_stream_end_time - llm_stream_start_time:.3f}s."
                         logger.error(llm_fail_msg)
                         current_log = _update_log(llm_fail_msg + f" Error: {chunk}", current_log)
                         err_state = current_chat_history[:-1] + [{"role": "assistant", "content": final_bot_message_text_clean}]
                         yield { chatbot_state: err_state, chatbot_display: err_state, log_state: current_log, log_output: current_log }; break

                    llm_text_buffer += chunk
                    disp_content = THINKING_MSG if not first_llm_chunk_processed else f"{LLM_RESPONSE_PREFIX}{llm_text_buffer}"
                    disp_hist = current_chat_history[:-1] + [{"role": "assistant", "content": disp_content}]
                    yield { chatbot_display: disp_hist, log_state: current_log, log_output: current_log }


                if not llm_error:
                    llm_stream_end_time = time.time()
                    llm_end_msg = f"<<< LLM API: Stream finished in {llm_stream_end_time - llm_stream_start_time:.3f}s."
                    logger.info(llm_end_msg)
                    current_log = _update_log(llm_end_msg, current_log)
                    final_bot_message_text_clean = llm_text_buffer.strip()
                    logger.info(f"--- LLM Response: '{final_bot_message_text_clean[:100]}...'")

                    llm_state = current_chat_history[:-1] + [{"role": "assistant", "content": final_bot_message_text_clean}]
                    llm_disp = llm_state[:-1] + [{"role": "assistant", "content": f"{LLM_RESPONSE_PREFIX}{final_bot_message_text_clean}"}]
                    yield { chatbot_state: llm_state, chatbot_display: llm_disp, log_state: current_log, log_output: current_log }

                    if final_bot_message_text_clean:
                        tts_start_msg = ">>> TTS Pipeline: Initiating TTS stream for LLM response..."
                        logger.info(tts_start_msg)
                        current_log = _update_log(tts_start_msg, current_log)
                        yield { log_state: current_log, log_output: current_log }

                        tts_pipeline_start_time = time.time()
                        audio_gen = generate_speech_stream(
                            final_bot_message_text_clean, chosen_voice, tts_temperature, tts_top_p, tts_repetition_penalty,
                            buffer_groups_param=tts_buffer_groups, padding_ms_param=tts_padding_ms
                        )
                        stream_ok, first_yield = False, False
                        first_audio_yield_time = None

                        for sr, audio_chunk in audio_gen:
                            if not first_yield and audio_chunk is not None and audio_chunk.size > 0 :
                                first_audio_yield_time = time.time()
                                ttfa_msg = f"--- TTS Pipeline: First audio chunk ready after {first_audio_yield_time - tts_pipeline_start_time:.3f}s."
                                logger.info(ttfa_msg)
                                current_log = _update_log(ttfa_msg, current_log)
                                yield { streaming_audio: (sr, audio_chunk), log_state: current_log, log_output: current_log }
                                stream_ok = True
                            elif audio_chunk is not None and audio_chunk.size > 0:
                                yield { streaming_audio: (sr, audio_chunk) }
                                stream_ok = True
                            first_yield = True

                        tts_pipeline_end_time = time.time()
                        tts_end_msg = f"<<< TTS Pipeline: LLM+TTS audio finished in {tts_pipeline_end_time - tts_pipeline_start_time:.3f}s (TTS part only)."
                        logger.info(tts_end_msg)
                        current_log = _update_log(tts_end_msg, current_log)
                        yield { log_state: current_log, log_output: current_log }

                        if not stream_ok: logger.warning("--- Warning: TTS failed after successful LLM."); final_bot_message_text_clean += f"\n\n{TTS_FAILED_MSG}"
                        if not first_yield: final_yield_dict[streaming_audio] = None
                    else: logger.warning("--- Warning: LLM generated empty response, skipping TTS."); final_yield_dict[streaming_audio] = None

        # --- Branch C: Plain LLM ---
        elif effective_mode == "plain":
            plain_mode_msg = ">>> LLM Pipeline: Starting Plain LLM Mode..."
            logger.info(plain_mode_msg)
            current_log = _update_log(plain_mode_msg, current_log)
            yield { log_state: current_log, log_output: current_log }

            if not text_to_process.strip(): final_bot_message_text_clean = EMPTY_REQUEST_MSG
            else:
                llm_history = clean_chat_history(current_chat_history[:-2])
                llm_messages = [{"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}] + llm_history + [{"role": "user", "content": text_to_process}]
                llm_payload = {"messages": llm_messages}
                gen_params = { "lmstudio_temperature": lmstudio_temperature, "lmstudio_top_p": lmstudio_top_p, "lmstudio_max_new_tokens": lmstudio_max_new_tokens, "lmstudio_repetition_penalty": lmstudio_repetition_penalty, "lmstudio_top_k": lmstudio_top_k }

                llm_text_buffer = ""; llm_error = False; first_llm_chunk_processed = False
                llm_stream_start_time = time.time()
                llm_start_msg = ">>> LLM API: Initiating stream..."
                logger.info(llm_start_msg)
                current_log = _update_log(llm_start_msg, current_log)
                yield { log_state: current_log, log_output: current_log }
                llm_gen = call_lmstudio_streaming(llm_payload, gen_params)

                for chunk in llm_gen:
                    if not first_llm_chunk_processed and chunk is not None:
                         first_token_time = time.time()
                         llm_first_token_msg = f"--- LLM API: First token received after {first_token_time - llm_stream_start_time:.3f}s."
                         logger.info(llm_first_token_msg)
                         current_log = _update_log(llm_first_token_msg, current_log)
                         first_llm_chunk_processed = True

                    if chunk.startswith(LLM_FAILED_PREFIX):
                        final_bot_message_text_clean = chunk; llm_error = True
                        llm_stream_end_time = time.time()
                        llm_fail_msg = f"<<< LLM API: Stream ended with error after {llm_stream_end_time - llm_stream_start_time:.3f}s."
                        logger.error(llm_fail_msg)
                        current_log = _update_log(llm_fail_msg + f" Error: {chunk}", current_log)
                        err_state = current_chat_history[:-1] + [{"role": "assistant", "content": final_bot_message_text_clean}]
                        yield { chatbot_state: err_state, chatbot_display: err_state, log_state: current_log, log_output: current_log }; break

                    llm_text_buffer += chunk
                    disp_content = THINKING_MSG if not first_llm_chunk_processed else f"{LLM_RESPONSE_PREFIX}{llm_text_buffer}"
                    disp_hist = current_chat_history[:-1] + [{"role": "assistant", "content": disp_content}]
                    yield { chatbot_display: disp_hist, log_state: current_log, log_output: current_log }

                if not llm_error:
                    llm_stream_end_time = time.time()
                    llm_end_msg = f"<<< LLM API: Stream finished in {llm_stream_end_time - llm_stream_start_time:.3f}s."
                    logger.info(llm_end_msg)
                    current_log = _update_log(llm_end_msg, current_log)
                    yield { log_state: current_log, log_output: current_log }
                    final_bot_message_text_clean = llm_text_buffer.strip()
                    logger.info(f"--- LLM Response: '{final_bot_message_text_clean[:100]}...'")

            final_yield_dict[streaming_audio] = None

    except Exception as e:
        pipeline_end_time = time.time()
        error_msg_fmt = f"<<< Pipeline Error: Unexpected error after {pipeline_end_time - pipeline_start_time:.3f}s."
        logger.exception(error_msg_fmt)
        current_log = _update_log(error_msg_fmt + f" Details: {e}", current_log)
        final_bot_message_text_clean = UNEXPECTED_ERROR_MSG_FORMAT.format(e)
        final_yield_dict[streaming_audio] = None
        yield { log_state: current_log, log_output: current_log }


    # --- 7. Final State Update ---
    final_bot_message_display = final_bot_message_text_clean
    if effective_mode in ["llm", "plain"] and final_bot_message_text_clean and not final_bot_message_text_clean.startswith(LLM_FAILED_PREFIX) and not final_bot_message_text_clean.startswith("("):
         final_bot_message_display = f"{LLM_RESPONSE_PREFIX}{final_bot_message_text_clean}"

    if current_chat_history: current_chat_history[-1]["content"] = final_bot_message_text_clean
    else: logger.error("--- Error: Chat history empty before final update."); current_chat_history.append({"role": "assistant", "content": final_bot_message_text_clean})

    final_display_history = list(current_chat_history)
    if final_display_history: final_display_history[-1]["content"] = final_bot_message_display

    process_end_time = time.time()
    final_summary_msg = f"--- Processing Finished --- Total: {process_end_time - process_start_time:.3f}s (STT: {stt_duration:.3f}s) ---"
    logger.info(final_summary_msg)
    current_log = _update_log(final_summary_msg, current_log)

    logger.debug("--- UI: Yielding final chat state update.")
    final_yield_dict.update({
        chatbot_state: current_chat_history, chatbot_display: final_display_history,
        text_input_box: gr.update(), audio_input_mic: gr.update(),
        log_state: current_log, log_output: current_log # Include final log update
    })
    if streaming_audio not in final_yield_dict: final_yield_dict[streaming_audio] = None
    yield final_yield_dict


    # --- 8. Cleanup --bot.png0- (No cleanup currently active)
    # if audio_filepath_to_clean and os.path.exists(audio_filepath_to_clean):
    #     pass # Keep the file


# =============================================================================
# Gradio Interface Definition
# =============================================================================
logger.info("--- Setting up Gradio Interface ---")

# Custom theme with improved styling
theme_to_use = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_50",
)

with gr.Blocks(theme=theme_to_use, title="Streaming Orpheus Edge Chat Locally") as demo:
    # Header Section
    with gr.Row():
        gr.Markdown("""
        # 🎤 Orpheus Edge - Local Only Chat: Test Mule
        ### Voice-enabled local AI assistant with streaming - running 100% local chat
        """)

    # Model Info Bar
    with gr.Row(variant="panel"):
        gr.Markdown(f"""
        **LLM:** `{LMSTUDIO_MODEL}`  
        **TTS:** `{TTS_MODEL}`  
        **STT:** `{WHISPER_MODEL_NAME}`  
        **Server:** `{SERVER_BASE_URL}`  
        **CODEC:** `{'SNAC 0.98 kbps, 24 kHz, 19.8 M Params'}`
        """)

    # State variables
    chatbot_state = gr.State([])
    selected_mode = gr.State("tts")
    log_state = gr.State("")

    # Main Interface
    with gr.Row():
        # Left Column - Chat Display
        with gr.Column(scale=2):
            with gr.Group():
                chatbot_display = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    render_markdown=True,
                    avatar_images=(
                        "assets/user.png",  # User avatar (ensure this path exists)
                        "assets/bot.png"    # Bot avatar (ensure this path exists)
                    ),
                    type="messages" 
                )
                streaming_audio = gr.Audio(
                    label="Synthesized Speech",
                    streaming=True,
                    autoplay=True,
                    visible=True,
                    interactive=False,
                    waveform_options={"waveform_progress_color": "#3b82f6"}
                )

            with gr.Accordion("📄 Processing Log", open=False):
                log_output = gr.Textbox(
                    label="", 
                    lines=8,
                    max_lines=20, 
                    interactive=False,
                    container=False
                )

        # Right Column - Controls
        with gr.Column(scale=1):
            # Input Section
            with gr.Group():
                text_input_box = gr.Textbox(
                    label="Message",
                    placeholder="Type your message or use command tags like @tara-llm...",
                    lines=11,
                    max_lines=11,
                    container=False
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input_mic = gr.Audio(
                            label="Audio Input",
                            sources=["microphone", "upload"],
                            type="filepath",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            tts_btn = gr.Button("🎙️ TTS my Text", variant="primary", size="sm")
                            llm_tts_btn = gr.Button("💬 LLM+TTS Chat", variant="secondary", size="sm")
                            plain_btn = gr.Button("📝 LLM Text Chat", variant="secondary", size="sm")
                        submit_button = gr.Button("Send", variant="primary")
                        clear_button = gr.Button("Clear", variant="stop", size="sm")
          
            with gr.Accordion("🔊 Streaming Settings", open=False):
                with gr.Row():
                    tts_buffer_input = gr.Slider( 
                        label="Buffer Size (Groups)", 
                        minimum=5,
                        maximum=80,
                        step=1,
                        value=TTS_STREAM_MIN_GROUPS,
                        info="Lower = faster start, shorter chunks"
                    )
                    tts_padding_input = gr.Slider( 
                        label="Silence Padding (ms)",
                        minimum=5,
                        maximum=200,
                        step=5,
                        value=TTS_STREAM_SILENCE_MS,
                        info="Adds silence between chunks"
                    )

            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                with gr.Tab("LLM Settings"):
                    lmstudio_max_new_tokens_slider = gr.Slider(
                        label="Max New Tokens",
                        minimum=-1, maximum=MAX_MAX_NEW_TOKENS, step=32, value=DEFAULT_LMSTUDIO_MAX_TOKENS,
                        info="-1 = server default"
                    )
                    with gr.Row():
                        lmstudio_temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=DEFAULT_LMSTUDIO_TEMP)
                        lmstudio_top_p_slider = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_LMSTUDIO_TOP_P)
                    with gr.Row():
                        lmstudio_top_k_slider = gr.Slider(label="Top-k", minimum=0, maximum=100, step=1, value=DEFAULT_LMSTUDIO_TOP_K)
                        lmstudio_repetition_penalty_slider = gr.Slider(label="Rep Penalty", minimum=1.0, maximum=2.0, step=0.05, value=DEFAULT_LMSTUDIO_REP_PENALTY)

                with gr.Tab("TTS Settings"):
                    with gr.Row():
                        tts_temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=DEFAULT_TTS_TEMP)
                        tts_top_p_slider = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_TTS_TOP_P)
                    tts_repetition_penalty_slider = gr.Slider(label="Rep Penalty", minimum=1.0, maximum=2.0, step=0.05, value=DEFAULT_TTS_REP_PENALTY)

    # =============================================================================
    # Gradio Event Handlers
    # =============================================================================

    def set_mode_tts():
        logger.info("UI Mode selected: TTS")
        return { tts_btn: gr.update(variant="primary"), llm_tts_btn: gr.update(variant="secondary"), plain_btn: gr.update(variant="secondary"), selected_mode: "tts" }
    def set_mode_llm():
        logger.info("UI Mode selected: LLM+TTS")
        return { tts_btn: gr.update(variant="secondary"), llm_tts_btn: gr.update(variant="primary"), plain_btn: gr.update(variant="secondary"), selected_mode: "llm" }
    def set_mode_plain():
        logger.info("UI Mode selected: Plain LLM")
        return { tts_btn: gr.update(variant="secondary"), llm_tts_btn: gr.update(variant="secondary"), plain_btn: gr.update(variant="primary"), selected_mode: "plain" }
    mode_outputs = [tts_btn, llm_tts_btn, plain_btn, selected_mode]
    tts_btn.click(fn=set_mode_tts, inputs=None, outputs=mode_outputs, api_name=False, queue=False)
    llm_tts_btn.click(fn=set_mode_llm, inputs=None, outputs=mode_outputs, api_name=False, queue=False)
    plain_btn.click(fn=set_mode_plain, inputs=None, outputs=mode_outputs, api_name=False, queue=False)


    def handle_submit(*args: Any):
        # *** The last argument is now the log_state ***
        log_state_val = args[-1]
        chat_hist = args[-2] # Chat history is second to last
        input_args = args[:-2]  # All other arguments are the inputs
        yield from process_input_blocks(*input_args, chat_history_value=chat_hist, log_state_value=log_state_val)

    all_param_inputs = [
        lmstudio_max_new_tokens_slider, lmstudio_temperature_slider, lmstudio_top_p_slider,
        lmstudio_top_k_slider, lmstudio_repetition_penalty_slider,
        tts_temperature_slider, tts_top_p_slider, tts_repetition_penalty_slider,
        tts_buffer_input, # Now a Slider
        tts_padding_input, # Now a Slider
    ]
    all_main_inputs = [
        text_input_box, audio_input_mic, selected_mode,
    ] + all_param_inputs + [
        chatbot_state,
        log_state      # Log state must be last
    ]
    # *** UPDATED outputs to include log components ***
    all_main_outputs = [
        chatbot_state, chatbot_display, text_input_box, audio_input_mic, streaming_audio,
        log_state, log_output # Add log state and output textbox
    ]

    submit_button.click(fn=handle_submit, inputs=all_main_inputs, outputs=all_main_outputs, api_name="chat_submit")
    text_input_box.submit(fn=handle_submit, inputs=all_main_inputs, outputs=all_main_outputs, api_name="chat_enter")

    # --- Clear Chat Handler (also clears UI log) ---
    def clear_chat():
        logger.info("Clearing chat history and inputs.")
        # *** Also clear the log state and output ***
        return {
            chatbot_state: [], chatbot_display: [], text_input_box: "",
            audio_input_mic: None, streaming_audio: None,
            log_state: "", log_output: "" # Clear log components
        }
    clear_button.click(fn=clear_chat, inputs=None, outputs=all_main_outputs, api_name=False, queue=False)


# =============================================================================
# Application Entry Point & Launch
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info(f"Launching Gradio {gr.__version__} Interface")
    logger.info(f"Mode: Client-Server")
    logger.info("-" * 60)
    logger.info(f"Server Base URL: {SERVER_BASE_URL}")
    logger.info(f" -> LLM Endpoint: {LMSTUDIO_API_ENDPOINT} (Model: {LMSTUDIO_MODEL})")
    logger.info(f" -> TTS Endpoint: {TTS_API_ENDPOINT} (Model: {TTS_MODEL})")
    logger.info("-" * 60)

    if whisper_model: logger.info(f"Local STT Model: Whisper '{WHISPER_MODEL_NAME}' loaded successfully on '{stt_device}'. Audio input enabled.")
    else: logger.warning("Local STT Model (Whisper) FAILED to load. Audio input will be disabled.")
    if snac_model: logger.info(f"Local Vocoder Model: SNAC loaded successfully on '{tts_device}'. TTS output enabled.")
    else: logger.critical("Local Vocoder Model (SNAC) FAILED to load. TTS output will likely fail.")
    logger.info("=" * 60)

    try:
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
        logger.info(f"Temporary audio directory '{TEMP_AUDIO_DIR}' is ready.")
    except OSError as e:
        logger.error(f"Could not create temporary audio directory '{TEMP_AUDIO_DIR}': {e}")

    share_enabled = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    server_port = int(os.getenv("GRADIO_PORT", 7860))
    logger.info(f"Starting Gradio server on 0.0.0.0:{server_port} (Share={share_enabled})")

    # Ensure assets folder exists for avatars (or remove avatar_images from Chatbot)
    if not os.path.exists("assets"):
        logger.warning("Assets folder not found. Creating 'assets' directory.")
        try: os.makedirs("assets")
        except OSError as e: logger.error(f"Could not create assets dir: {e}")
    # Add dummy avatar files if they don't exist, to prevent errors
    if not os.path.exists("assets/user.png"): logger.warning("User avatar 'assets/user.png' not found. Chatbot may display default.")
    if not os.path.exists("assets/bot.png"): logger.warning("Bot avatar 'assets/bot.png' not found. Chatbot may display default.")


# ================
# 🚀 Run the App
# ================
if __name__ == "__main__":
    try:
        demo.queue().launch(
            share=False,
            server_name="0.0.0.0",
            server_port=server_port,
            inbrowser=True
        )
       
    except KeyboardInterrupt:
        print("🛑 Server manually stopped by user.")
    except ConnectionResetError:
        print("⚠️ Connection was reset by browser, safe to ignore.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

logger.info("Gradio Interface Launched and Running.")
