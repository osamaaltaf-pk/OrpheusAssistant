# -*- coding: utf-8 -*-
# =============================================================================
# Imports
# =============================================================================
import os
import random
import uuid
import json
import time
import traceback
import re
import gradio as gr
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import requests
from dotenv import load_dotenv
import subprocess

# =============================================================================
# Library Imports with Error Handling
# =============================================================================
# FFmpeg Import
try:
    import ffmpeg
    print("ffmpeg-python library imported successfully.")
except ImportError:
    print("Warning: ffmpeg-python not found. Falling back to subprocess.")
    ffmpeg = None

# Whisper Import
try:
    import whisper
    print("Whisper library imported successfully.")
except ImportError:
    print("ERROR: Whisper library not found. Please install it:")
    print("pip install -U openai-whisper")
    print("Ensure ffmpeg is also installed and in your system PATH.")
    exit(1)

# SNAC Import
try:
    from snac import SNAC
except ImportError:
    print("ERROR: SNAC library not found. Please install it:")
    print("pip install git+https://github.com/hubertsiuzdak/snac.git")
    exit(1)

# =============================================================================
# Configuration
# =============================================================================
# Load environment variables
load_dotenv()

# Unified Server Configuration
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")

# LM Studio Configuration
LMSTUDIO_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/chat/completions"
LMSTUDIO_MODEL = "dolphin3.0-llama3.1-8b-abliterated"  # Updated model name
LMSTUDIO_SYSTEM_PROMPT = (
    "You are in very funny chat with consenting adults and you are an impatient "
    "midwestern 32 year old who just wants the code from her chat partner. "
    "You respond in mildly profane and sarcastic single sentences about python code"
)

# TTS Configuration
TTS_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/completions"
TTS_MODEL = "orpheus-3b-ft.gguf"

# =============================================================================
# Device Setup
# =============================================================================
if torch.cuda.is_available():
    tts_device = "cuda"
    stt_device = "cuda"
    print("SNAC vocoder and Whisper STT will use CUDA if possible.")
else:
    tts_device = "cpu"
    stt_device = "cpu"
    print("CUDA not available. SNAC vocoder and Whisper STT will use CPU.")

# =============================================================================
# Model Loading
# =============================================================================
print("Loading SNAC vocoder model...")
snac_model = None
try:
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(tts_device)
    snac_model.eval()
    print(f"SNAC vocoder loaded to {tts_device}")
except Exception as e:
    print(f"Error loading SNAC model: {e}")

print("Loading Whisper STT model...")
WHISPER_MODEL_NAME = "base.en"
whisper_model = None
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=stt_device)
    print(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

# =============================================================================
# Constants
# =============================================================================
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_LMSTUDIO_MAX_TOKENS = -1 #See LM Studo Model config
MAX_SEED = np.iinfo(np.int32).max
ORPHEUS_MIN_ID = 10
ORPHEUS_TOKENS_PER_LAYER = 4096
ORPHEUS_N_LAYERS = 7
ORPHEUS_MAX_ID = ORPHEUS_MIN_ID + (ORPHEUS_N_LAYERS * ORPHEUS_TOKENS_PER_LAYER)

# Default parameters
DEFAULT_LMSTUDIO_TEMP = 0.7
DEFAULT_LMSTUDIO_TOP_P = 0.9
DEFAULT_LMSTUDIO_TOP_K = 40
DEFAULT_LMSTUDIO_REP_PENALTY = 1.1

DEFAULT_TTS_TEMP = 0.8
DEFAULT_TTS_TOP_P = 0.9
DEFAULT_TTS_TOP_K = 40
DEFAULT_TTS_REP_PENALTY = 1.1

# Conversation History
CONTEXT_TURN_LIMIT = 3

# =============================================================================
# Utility Functions
# =============================================================================
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def clean_chat_history(limited_chat_history):
    """Clean chat history for LLM input."""
    cleaned_lmstudio_format = []
    if not limited_chat_history:
        return []
        
    for user_msg_display, bot_msg_display in limited_chat_history:
        user_text = None
        if isinstance(user_msg_display, str):
            if user_msg_display.startswith("🎤 (Audio Input): "):
                user_text = user_msg_display.split("🎤 (Audio Input): ", 1)[1]
            elif user_msg_display.startswith(("@tara-tts ", "@tara-llm ")):
                user_text = user_msg_display.split(" ", 1)[1]
            else:
                user_text = user_msg_display
        elif isinstance(user_msg_display, tuple):
            if len(user_msg_display) > 1 and isinstance(user_msg_display[1], str):
                user_text = user_msg_display[1].replace("🎤: ", "")
            elif isinstance(user_msg_display[0], str) and not user_msg_display[0].endswith((".wav", ".mp3")):
                user_text = user_msg_display[0]
        
        bot_text = None
        if isinstance(bot_msg_display, tuple):
            if len(bot_msg_display) > 1 and isinstance(bot_msg_display[1], str):
                bot_text = bot_msg_display[1]
        elif isinstance(bot_msg_display, str):
            if not bot_msg_display.startswith(("[Error", "(Error", "Sorry,", "(No input", "Processing", "(TTS failed")):
                bot_text = bot_msg_display
        
        if user_text and user_text.strip():
            cleaned_lmstudio_format.append({"role": "user", "content": user_text})
        if bot_text and bot_text.strip():
            cleaned_lmstudio_format.append({"role": "assistant", "content": bot_text})
            
    return cleaned_lmstudio_format

def save_audio_as_mp3(audio_data: np.ndarray, sample_rate: int) -> str:
    """Save numpy audio array as MP3 file (with WAV fallback)."""
    temp_dir = "temp_audio_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    temp_path = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4().hex}.mp3")
    temp_wav_path = temp_path.replace('.mp3', '.wav')
    
    # 1. First save as WAV (required for ffmpeg input)
    wavfile.write(temp_wav_path, sample_rate, audio_data)
    
    # 2. Convert to MP3 for later use with meme tools
    try:
        if ffmpeg is not None:
            # Using ffmpeg-python if available
            (
                ffmpeg.input(temp_wav_path)
                .output(temp_path, audio_bitrate='192k')
                .overwrite_output()
                .run(quiet=True)
            )
        else:
            # Fallback to subprocess
            subprocess.run([
                'ffmpeg', '-i', temp_wav_path,
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',  # Quality setting (2=high)
                '-y',  # Overwrite without asking
                temp_path
            ], check=True)
        
        print(f"Saved MP3 audio: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"MP3 conversion failed, using WAV instead. Error: {e}")
        return temp_wav_path  # Fallback to original WAV

# =============================================================================
# TTS Pipeline Functions
# =============================================================================
def parse_gguf_codes(response_text): #debug entry point
    """Parse GGUF token IDs from LLM response."""
    return [
        int(match) for match in re.findall(r"<custom_token_(\d+)>", response_text)
        if ORPHEUS_MIN_ID <= int(match) < ORPHEUS_MAX_ID
    ]

def redistribute_codes(absolute_code_list, target_snac_model):
    """Convert GGUF tokens to SNAC-compatible format and generate audio."""
    if not absolute_code_list or target_snac_model is None:
        return None
    
    snac_device = next(target_snac_model.parameters()).device
    layer_1, layer_2, layer_3 = [], [], []
    num_tokens = len(absolute_code_list)
    num_groups = num_tokens // ORPHEUS_N_LAYERS
    
    if num_groups == 0:
        return None
    
    print(f"  - Processing {num_groups} groups of {ORPHEUS_N_LAYERS} codes for SNAC...")
    
    for i in range(num_groups):
        base_idx = i * ORPHEUS_N_LAYERS
        if base_idx + ORPHEUS_N_LAYERS > num_tokens:
            break
            
        group_codes = absolute_code_list[base_idx:base_idx + ORPHEUS_N_LAYERS]
        processed_group = [None] * ORPHEUS_N_LAYERS
        valid_group = True
        
        for j, token_id in enumerate(group_codes):
            if not (ORPHEUS_MIN_ID <= token_id < ORPHEUS_MAX_ID):
                valid_group = False
                break
                
            layer_index = (token_id - ORPHEUS_MIN_ID) // ORPHEUS_TOKENS_PER_LAYER
            code_index = (token_id - ORPHEUS_MIN_ID) % ORPHEUS_TOKENS_PER_LAYER
            
            if layer_index != j:
                valid_group = False
                break
                
            processed_group[j] = code_index
            
        if not valid_group:
            continue
            
        try:
            layer_1.append(processed_group[0])
            layer_2.append(processed_group[1])
            layer_3.append(processed_group[2])
            layer_3.append(processed_group[3])
            layer_2.append(processed_group[4])
            layer_3.append(processed_group[5])
            layer_3.append(processed_group[6])
        except (IndexError, TypeError):
            continue
    
    try:
        if not layer_1 or not layer_2 or not layer_3:
            return None
            
        print(f"  - Final SNAC layer sizes: L1={len(layer_1)}, L2={len(layer_2)}, L3={len(layer_3)}")
        
        codes = [
            torch.tensor(layer_1, device=snac_device, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_2, device=snac_device, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_3, device=snac_device, dtype=torch.long).unsqueeze(0)
        ]
        
        with torch.no_grad():
            audio_hat = target_snac_model.decode(codes)
            
        return audio_hat.detach().squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error during tensor creation or SNAC decoding: {e}")
        return None

def generate_speech_gguf(text, voice, tts_temperature, tts_top_p, tts_repetition_penalty, max_new_tokens_audio):
    """Generate speech from text using GGUF model and SNAC vocoder."""
    if not text.strip() or snac_model is None:
        return None
        
    print(f"Generating speech via TTS server for: '{text[:50]}...'")
    start_time = time.time()
    
    payload = {
        "model": TTS_MODEL,
        "prompt": f"<|audio|>{voice}: {text}<|eot_id|>",
        "temperature": tts_temperature,
        "top_p": tts_top_p,
        "repeat_penalty": tts_repetition_penalty,
        "max_tokens": max_new_tokens_audio,
        "stop": ["<|eot_id|>", "<|audio|>"],
        "stream": False
    }
    
    print(f"  - Sending payload to {TTS_API_ENDPOINT} (Model: {TTS_MODEL})")
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            TTS_API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=180
        )
        response.raise_for_status()
        response_json = response.json()
        
        print(f"  - Raw TTS response: {json.dumps(response_json, indent=2)[:200]}...")
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            raw_generated_text = response_json["choices"][0].get("text", "").strip()
            if not raw_generated_text:
                print("Error: Empty text in TTS response")
                return None
                
            req_time = time.time()
            print(f"  - TTS server request took {req_time - start_time:.2f}s")
            
            absolute_id_list = parse_gguf_codes(raw_generated_text)
            if not absolute_id_list:
                print("Error: No valid audio codes parsed. Raw text:", raw_generated_text[:200])
                return None
                
            audio_samples = redistribute_codes(absolute_id_list, snac_model)
            if audio_samples is None:
                print("Error: Failed to generate audio samples from tokens")
                return None
                
            snac_time = time.time()
            print(f"  - Generated audio samples via SNAC, shape: {audio_samples.shape}")
            print(f"  - Total TTS generation time: {snac_time - start_time:.2f}s")
            return (24000, audio_samples)
            
        else:
            print(f"Error: Unexpected TTS response format: {response_json}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error during request to TTS server: {e}")
        return None
    except Exception as e:
        print(f"Error during TTS generation pipeline: {e}")
        traceback.print_exc()
        return None

# =============================================================================
# LLM Communication -  TTS part needs complete text first (Non Streaming optimized)
# =============================================================================
def call_lmstudio_non_streaming(lmstudio_payload, generation_params):
    """Call LM Studio API for non-streaming response."""
    final_response = "[Error: Default response]"
    try:
        payload = {
            "model": LMSTUDIO_MODEL,
            "messages": lmstudio_payload["messages"],
            "temperature": generation_params.get('lmstudio_temperature', DEFAULT_LMSTUDIO_TEMP),
            "top_p": generation_params.get('lmstudio_top_p', DEFAULT_LMSTUDIO_TOP_P),
            "max_tokens": generation_params.get('lmstudio_max_new_tokens', DEFAULT_LMSTUDIO_MAX_TOKENS),
            "repeat_penalty": generation_params.get('lmstudio_repetition_penalty', DEFAULT_LMSTUDIO_REP_PENALTY),
            "stream": False
        }
        
        print(f"  - Sending to {LMSTUDIO_API_ENDPOINT} with model {LMSTUDIO_MODEL}")
        
        headers = {"Content-Type": "application/json"}
        start_time = time.time()
        response = requests.post(
            LMSTUDIO_API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=180
        )
        response.raise_for_status()
        response_json = response.json()
        end_time = time.time()
        
        print(f"  - LLM request took {end_time - start_time:.2f}s")
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "message" in choice:
                final_response = choice["message"]["content"].strip()
            elif "text" in choice:
                final_response = choice["text"].strip()
            else:
                final_response = "[Error: Unexpected response format]"
        else:
            final_response = f"[Error: {response_json.get('error', 'Unknown error')}]"
            
    except requests.exceptions.RequestException as e:
        final_response = f"[Error connecting to LLM: {e}]"
    except Exception as e:
        final_response = f"[Unexpected Error: {e}]"
        traceback.print_exc()
        
    print(f"  - LLM response: '{final_response[:100]}...'")
    return final_response

# =============================================================================
# Main Processing Function (Router)
# =============================================================================
def process_input_blocks(
    text_input: str, 
    audio_input_path: str,
    mode: str,  # "tts", "llm", or "plain"
    lmstudio_max_new_tokens: int,
    lmstudio_temperature: float,
    lmstudio_top_p: float,
    lmstudio_top_k: int,
    lmstudio_repetition_penalty: float,
    tts_temperature: float,
    tts_top_p: float,
    tts_repetition_penalty: float,
    chat_history: list
):
    """Main processing function for handling user input and generating responses."""
    global whisper_model, snac_model
    
    # Initialize variables
    original_user_input_text = ""
    user_display_input = None
    text_to_process = ""
    transcription_source = "text"
    bot_response = ""
    bot_audio_tuple = None
    audio_filepath_to_clean = None
    is_purely_text_input = False
    prefix_to_add = None
    force_plain_llm = False

    # Determine mode
    if mode == "tts":
        prefix_to_add = "@tara-tts"
        force_plain_llm = False
    elif mode == "llm":
        prefix_to_add = "@tara-llm"
        force_plain_llm = False
    else:  # plain
        prefix_to_add = None
        force_plain_llm = True

    # Handle Audio Input
    if audio_input_path and whisper_model:
        if os.path.isfile(audio_input_path):
            audio_filepath_to_clean = audio_input_path
            transcription_source = "voice"
            print(f"Processing audio input: {audio_input_path}")
            try:
                stt_start_time = time.time()
                result = whisper_model.transcribe(audio_input_path, fp16=(stt_device == 'cuda'))
                original_user_input_text = result["text"].strip()
                stt_end_time = time.time()
                print(f"  - Whisper transcription: '{original_user_input_text}' (took {stt_end_time - stt_start_time:.2f}s)")
                user_display_input = f"🎤 (Audio Input): {original_user_input_text}"
                text_to_process = original_user_input_text

                # Check if transcription is already a command
                known_prefixes = ["@tara-tts", "@jess-tts", "@leo-tts", "@leah-tts", "@dan-tts", "@mia-tts", "@zac-tts", "@zoe-tts",
                                "@tara-llm", "@jess-llm", "@leo-llm", "@leah-llm", "@dan-llm", "@mia-llm", "@zac-llm", "@zoe-llm"]
                is_already_command = any(original_user_input_text.lower().startswith(p) for p in known_prefixes)

                if not is_already_command:
                    if force_plain_llm:
                        prefix_to_add = None
                        print(f"  - Plain LLM mode. Processing audio as text input for LLM.")
                    elif prefix_to_add:
                        text_to_process = f"{prefix_to_add} {original_user_input_text}"
                        print(f"  - Auto-prefix {prefix_to_add} applied to audio.")
                else:
                    print(f"  - Transcribed audio is already a command '{original_user_input_text[:20]}...'.")
                    text_to_process = original_user_input_text

            except Exception as e:
                print(f"Error during Whisper transcription: {e}")
                traceback.print_exc()
                error_msg = f"[Error during local transcription: {e}]"
                chat_history.append((f"🎤 (Audio Input Error: {audio_input_path})", error_msg))
                return chat_history, None, None
        else:
            print(f"Received invalid audio path: {audio_input_path}, falling back to text.")

    # Handle Text Input
    if not text_to_process and text_input:
        original_user_input_text = text_input.strip()
        user_display_input = original_user_input_text
        print(f"Processing text input: '{original_user_input_text}'")
        transcription_source = "text"
        text_to_process = original_user_input_text
        
        known_prefixes = ["@tara-tts", "@jess-tts", "@leo-tts", "@leah-tts", "@dan-tts", "@mia-tts", "@zac-tts", "@zoe-tts",
                         "@tara-llm", "@jess-llm", "@leo-llm", "@leah-llm", "@dan-llm", "@mia-llm", "@zac-llm", "@zoe-llm"]
        is_already_command = any(original_user_input_text.lower().startswith(p) for p in known_prefixes)
        
        if not is_already_command:
            if force_plain_llm:
                prefix_to_add = None
                print(f"  - Plain LLM mode. Processing text input for LLM.")
            elif prefix_to_add:
                text_to_process = f"{prefix_to_add} {original_user_input_text}"
                print(f"  - Auto-prefix {prefix_to_add} applied to text.")
        else:
            print(f"  - User provided command in text '{original_user_input_text[:20]}...', not auto-prepending.")
            text_to_process = original_user_input_text

    if not text_to_process:
        print("No valid text or audio input to process.")
        return chat_history, None, None

    chat_history.append((user_display_input, None))

    # Process Input Text
    lower_text = text_to_process.lower()
    print(f"  - Routing query ({transcription_source}): '{text_to_process[:100]}...'")
    
    all_voices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
    tts_tags = {f"@{voice}-tts": voice for voice in all_voices}
    llm_tags = {f"@{voice}-llm": voice for voice in all_voices}
    
    final_bot_message = None

    try:
        matched_tts = False
        matched_llm_tts = False

        # Check Branches
        if not force_plain_llm:
            # Branch 1: Direct TTS
            for tag, voice in tts_tags.items():
                if lower_text.startswith(tag):
                    matched_tts = True
                    text_to_speak = text_to_process[len(tag):].strip()
                    print(f"  - Direct TTS request for voice '{voice}': '{text_to_speak[:50]}...'")
                    if snac_model is None:
                        raise ValueError("SNAC vocoder not loaded.")
                    audio_output = generate_speech_gguf(
                        text_to_speak, voice, 
                        tts_temperature, tts_top_p, tts_repetition_penalty, 
                        MAX_MAX_NEW_TOKENS
                    )
                    if audio_output:
                        sample_rate, audio_data = audio_output
                        if audio_data.dtype != np.int16:
                            if np.issubdtype(audio_data.dtype, np.floating):
                                max_val = np.max(np.abs(audio_data))
                                audio_data = np.int16(audio_data/max_val*32767) if max_val > 1e-6 else np.zeros_like(audio_data, dtype=np.int16)
                            else:
                                audio_data = audio_data.astype(np.int16)
                        temp_audio_path = save_audio_as_mp3(audio_data, sample_rate)
                        print(f"  - Saved TTS audio: {temp_audio_path}")
                        final_bot_message = (temp_audio_path, None)
                    else:
                        final_bot_message = f"Sorry, couldn't generate speech for '{text_to_speak[:50]}...'."
                    break

            # Branch 2: LLM + TTS
            if not matched_tts:
                for tag, voice in llm_tags.items():
                    if lower_text.startswith(tag):
                        matched_llm_tts = True
                        prompt_for_llm = text_to_process[len(tag):].strip()
                        print(f"  - LLM+TTS request for voice '{voice}': '{prompt_for_llm[:75]}...'")
                        if snac_model is None:
                            raise ValueError("SNAC vocoder not loaded.")
                        
                        history_before_current = chat_history[:-1]
                        limited_history_turns = history_before_current[-CONTEXT_TURN_LIMIT:]
                        cleaned_hist_for_llm = clean_chat_history(limited_history_turns)
                        
                        messages = [
                            {"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}
                        ] + cleaned_hist_for_llm + [
                            {"role": "user", "content": prompt_for_llm}
                        ]
                        
                        llm_params = {
                            'lmstudio_temperature': lmstudio_temperature,
                            'lmstudio_top_p': lmstudio_top_p,
                            'lmstudio_top_k': lmstudio_top_k,
                            'lmstudio_max_new_tokens': lmstudio_max_new_tokens,
                            'lmstudio_repetition_penalty': lmstudio_repetition_penalty
                        }
                        
                        llm_response_text = call_lmstudio_non_streaming(
                            {"messages": messages},
                            llm_params
                        )
                        
                        if llm_response_text and not llm_response_text.startswith("[Error"):
                            audio_output = generate_speech_gguf(
                                llm_response_text, voice,
                                tts_temperature, tts_top_p, tts_repetition_penalty,
                                MAX_MAX_NEW_TOKENS
                            )
                            if audio_output:
                                sample_rate, audio_data = audio_output
                                if audio_data.dtype != np.int16:
                                    if np.issubdtype(audio_data.dtype, np.floating):
                                        max_val = np.max(np.abs(audio_data))
                                        audio_data = np.int16(audio_data/max_val*32767) if max_val > 1e-6 else np.zeros_like(audio_data, dtype=np.int16)
                                    else:
                                        audio_data = audio_data.astype(np.int16)
                                temp_audio_path = save_audio_as_mp3(audio_data, sample_rate)
                                print(f"  - Saved LLM+TTS audio: {temp_audio_path}")
                                final_bot_message = (temp_audio_path, llm_response_text)
                            else:
                                print("Warning: TTS generation failed...")
                                final_bot_message = f"{llm_response_text}\n\n(TTS failed...)"
                        else:
                            final_bot_message = llm_response_text
                        break

        # Branch 3: Plain LLM
        if force_plain_llm or (not matched_tts and not matched_llm_tts):
            if force_plain_llm:
                print(f"  - Plain LLM chat mode selected...")
            else:
                print(f"  - Default text chat (no command prefix detected/added)...")

            history_before_current = chat_history[:-1]
            limited_history_turns = history_before_current[-CONTEXT_TURN_LIMIT:]
            cleaned_hist_for_llm = clean_chat_history(limited_history_turns)
            
            messages = [
                {"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}
            ] + cleaned_hist_for_llm + [
                {"role": "user", "content": original_user_input_text}
            ]
            
            llm_params = {
                'lmstudio_temperature': lmstudio_temperature,
                'lmstudio_top_p': lmstudio_top_p,
                'lmstudio_top_k': lmstudio_top_k,
                'lmstudio_max_new_tokens': lmstudio_max_new_tokens,
                'lmstudio_repetition_penalty': lmstudio_repetition_penalty
            }
            
            final_bot_message = call_lmstudio_non_streaming(
                {"messages": messages},
                llm_params
            )

    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        final_bot_message = f"[An unexpected error occurred: {e}]"

    chat_history[-1] = (user_display_input, final_bot_message)
    return chat_history, None, None

# =============================================================================
# Gradio Interface
# =============================================================================
print("Setting up Gradio Interface with gr.Blocks...")
theme_to_use = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme_to_use) as demo:
    gr.Markdown(f"# Orpheus Edge 🎤 ({LMSTUDIO_MODEL}) Chat & TTS")
    
    chatbot = gr.Chatbot(label="Chat History", height=500)
    
    with gr.Row():
        with gr.Column(scale=3):
            text_input_box = gr.Textbox(label="Type your message or use microphone", lines=2)
        with gr.Column(scale=1):
            audio_input_mic = gr.Audio(label="Record Audio Input", type="filepath")
    
    # Mode selection buttons
    with gr.Row():
        tts_btn = gr.Button("🎙️ TTS Mode", variant="primary", size="sm")
        llm_tts_btn = gr.Button("💬 LLM+TTS Mode", variant="secondary", size="sm")
        plain_btn = gr.Button("📝 Text Mode", variant="secondary", size="sm")
    
    # Action buttons
    with gr.Row():
        submit_button = gr.Button("Send / Submit", variant="primary")
        clear_button = gr.Button("Clear", variant="stop")

    # Clear button handler 
    clear_button.click(
        fn=lambda: ([], None, None),
        outputs=[chatbot, text_input_box, audio_input_mic]
    )
    
    # Mode state tracker
    selected_mode = gr.State("tts")  # Default to TTS mode
    
    # Button click handlers with visual feedback
    def set_mode_tts():
        return [
            gr.Button(variant="primary"),  # Active (blue)
            gr.Button(variant="secondary"),  # Inactive
            gr.Button(variant="secondary"),  # Inactive
            "tts"  # selected_mode
        ]
    
    def set_mode_llm():
        return [
            gr.Button(variant="secondary"),
            gr.Button(variant="primary"),  # Active (blue)
            gr.Button(variant="secondary"),
            "llm"
        ]
    
    def set_mode_plain():
        return [
            gr.Button(variant="secondary"),
            gr.Button(variant="secondary"),
            gr.Button(variant="primary"),  # Active (blue)
            "plain"
        ]
    
    tts_btn.click(set_mode_tts, outputs=[tts_btn, llm_tts_btn, plain_btn, selected_mode])
    llm_tts_btn.click(set_mode_llm, outputs=[tts_btn, llm_tts_btn, plain_btn, selected_mode])
    plain_btn.click(set_mode_plain, outputs=[tts_btn, llm_tts_btn, plain_btn, selected_mode])

    with gr.Accordion("Generation Parameters", open=False):
        gr.Markdown("### LLM Parameters")
        lmstudio_max_new_tokens_slider = gr.Slider(label="Max New Tokens", minimum=32, maximum=4096, step=32, value=DEFAULT_LMSTUDIO_MAX_TOKENS)
        lmstudio_temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=DEFAULT_LMSTUDIO_TEMP)
        lmstudio_top_p_slider = gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=DEFAULT_LMSTUDIO_TOP_P)
        lmstudio_top_k_slider = gr.Slider(label="Top-k", minimum=1, maximum=100, step=1, value=DEFAULT_LMSTUDIO_TOP_K)
        lmstudio_repetition_penalty_slider = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=DEFAULT_LMSTUDIO_REP_PENALTY)
        
        gr.Markdown("---")
        gr.Markdown("### TTS Parameters")
        tts_temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=DEFAULT_TTS_TEMP)
        tts_top_p_slider = gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=DEFAULT_TTS_TOP_P)
        tts_repetition_penalty_slider = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=DEFAULT_TTS_REP_PENALTY)
    
    param_inputs = [
        lmstudio_max_new_tokens_slider, lmstudio_temperature_slider, lmstudio_top_p_slider,
        lmstudio_top_k_slider, lmstudio_repetition_penalty_slider,
        tts_temperature_slider, tts_top_p_slider, tts_repetition_penalty_slider
    ]
    
    # Update inputs list to include selected_mode
    all_inputs = [
        text_input_box,
        audio_input_mic,
        selected_mode,
    ] + param_inputs + [chatbot]
    
    submit_button.click(
        fn=process_input_blocks,
        inputs=all_inputs,
        outputs=[chatbot, text_input_box, audio_input_mic]
    )
    text_input_box.submit(
        fn=process_input_blocks,
        inputs=all_inputs,
        outputs=[chatbot, text_input_box, audio_input_mic]
    )

# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("-" * 50)
    print(f"Launching Gradio {gr.__version__} Interface")
    print(f"Whisper STT Model: {WHISPER_MODEL_NAME} on {stt_device}")
    print(f"SNAC Vocoder loaded to {tts_device}")
    print(f"Server URL: {SERVER_BASE_URL}")
    print(f"LLM Model: {LMSTUDIO_MODEL}")
    print(f"TTS Model: {TTS_MODEL}")
    print("-" * 50)
    print("Default Parameters:")
    print(f"  LLM: Temp={DEFAULT_LMSTUDIO_TEMP}, TopP={DEFAULT_LMSTUDIO_TOP_P}")
    print(f"  TTS: Temp={DEFAULT_TTS_TEMP}, TopP={DEFAULT_TTS_TOP_P}")
    print("-" * 50)
    print("Ensure your LM Studio server is running with both models loaded")
    os.makedirs("temp_audio_files", exist_ok=True)
    demo.launch(share=False)
    print("Gradio Interface launched. Press Ctrl+C to stop.")
