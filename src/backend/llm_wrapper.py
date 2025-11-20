"""
LLM Wrapper Module for Disaster Literacy RAG System
Supports offline (llama.cpp) and online (Google Gemini / OpenRouter) LLMs
"""

import os
from typing import Optional
from abc import ABC, abstractmethod

from config import (
    OFFLINE_LLM_MODEL_PATH_ECONOMY,
    OFFLINE_LLM_MODEL_PATH_POWER,
    OFFLINE_LLM_CONTEXT_LENGTH,
    OFFLINE_LLM_THREADS,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    OFFLINE_ECONOMY_MAX_LLM_TOKENS,
    OFFLINE_POWER_MAX_LLM_TOKENS,
    ONLINE_MAX_LLM_TOKENS,
    TEMPERATURE,
    LLM_TIMEOUT,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    DEFAULT_OFFLINE_MODE
)
from error_handler import error_handler, LLMInferenceError


class BaseLLM(ABC):
    """Base class for LLM implementations"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


# ==================== Offline LLM ====================

class OfflineLLM(BaseLLM):
    def __init__(self, model_mode="economy"):
        try:
            from llama_cpp import Llama

            # Select model based on mode
            self.model_mode = model_mode.lower()
            if self.model_mode == "power":
                model_path = OFFLINE_LLM_MODEL_PATH_POWER
                self.max_tokens = OFFLINE_POWER_MAX_LLM_TOKENS
            elif self.model_mode == "economy":
                model_path = OFFLINE_LLM_MODEL_PATH_ECONOMY
                self.max_tokens = OFFLINE_ECONOMY_MAX_LLM_TOKENS
            else:
                # Default to economy mode if mode is unknown
                error_handler.logger.warning(f"Unknown model mode '{model_mode}', defaulting to economy mode")
                model_path = OFFLINE_LLM_MODEL_PATH_ECONOMY
                self.max_tokens = OFFLINE_ECONOMY_MAX_LLM_TOKENS
                self.model_mode = "economy"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Offline model not found: {model_path}")

            error_handler.logger.info(f"Loading offline LLM ({self.model_mode} mode): {model_path}")
            error_handler.logger.info(f"Max tokens for {self.model_mode} mode: {self.max_tokens}")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=OFFLINE_LLM_CONTEXT_LENGTH,
                n_threads=OFFLINE_LLM_THREADS,
                n_gpu_layers=0,
                verbose=False,
            )

            error_handler.logger.info(f"Offline LLM loaded ({self.model_mode} mode)")

        except Exception as e:
            raise LLMInferenceError(f"Failed to load offline LLM: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Format prompt based on model type
        if self.model_mode == "power":
            # Qwen2 chat format: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            if system_prompt:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        else:
            # Llama-2-chat format: [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST]
            if system_prompt:
                full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                full_prompt = f"[INST] {prompt} [/INST]"
            stop_tokens = ["</s>", "[INST]", "[/INST]", "<s>"]
        
        try:
            response = self.llm(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=TEMPERATURE,
                stop=stop_tokens,
                echo=False
            )

            generated_text = response["choices"][0]["text"].strip()
            
            # Additional validation
            if not generated_text:
                error_handler.logger.warning("LLM returned empty text, checking full response")
                error_handler.logger.debug(f"Full response: {response}")
                raise LLMInferenceError("Empty response generated")
            
            return generated_text
        except Exception as e:
            error_handler.logger.error(f"Offline LLM generation failed: {e}")
            raise LLMInferenceError(f"Offline generation failed: {e}")


# ==================== Online LLM ====================

class OnlineLLM(BaseLLM):
    def __init__(self, provider: str = "google"):
        self.provider = provider.lower()
        self.model = None

        if self.provider == "google":
            self._init_google()
        elif self.provider == "openrouter":
            self._init_requests()
            self._init_openrouter()
        else:
            raise LLMInferenceError(f"Unknown provider: {provider}")

        error_handler.logger.info(f"Online LLM Ready: {self.provider} | Model: {self.model}")

    # --- Setup Methods ---
    def _init_google(self):
        if not GOOGLE_API_KEY:
            raise LLMInferenceError("Missing GOOGLE_API_KEY")
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        self.client = genai.GenerativeModel(GOOGLE_MODEL)
        self.model = GOOGLE_MODEL

    def _init_requests(self):
        import requests
        self.session = requests.Session()

    def _init_openrouter(self):
        if not OPENROUTER_API_KEY:
            raise LLMInferenceError("Missing OPENROUTER_API_KEY")
        self.or_api_key = OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL or "qwen/qwen-2.5-72b-instruct:free"

    # --- Generate ---
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.provider == "google":
            return self._google(prompt, system_prompt)
        if self.provider == "openrouter":
            return self._openrouter(prompt, system_prompt)
        raise LLMInferenceError("Unknown provider")

    # --- Google ---
    def _google(self, prompt, system_prompt):
        import time
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={"temperature": TEMPERATURE, "max_output_tokens": ONLINE_MAX_LLM_TOKENS}
                )
                return response.text.strip() if hasattr(response, "text") else ""
            except Exception as e:
                error_str = str(e)
                # Check for rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        import re
                        retry_match = re.search(r'retry in ([0-9.]+)s', error_str)
                        wait_time = float(retry_match.group(1)) if retry_match else 12
                        error_handler.logger.warning(f"Rate limit hit, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                # Re-raise if not rate limit or max retries exceeded
                raise

    # --- OpenRouter ---
    def _openrouter(self, prompt, system_prompt):
        import requests

        api_url = "https://openrouter.ai/api/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": ONLINE_MAX_LLM_TOKENS
        }

        headers = {
            "Authorization": f"Bearer {self.or_api_key}",
            "HTTP-Referer": "http://localhost",
            "Content-Type": "application/json",
            "X-Title": "Disaster RAG App"
        }

        r = self.session.post(api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT)

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()

        raise LLMInferenceError(f"OpenRouter ERROR {r.status_code}: {r.text}")


# ==================== Wrapper ====================

class LLMWrapper:
    def __init__(self, mode="offline", provider="google", offline_model_mode="economy"):
        self.mode = mode.lower()
        if self.mode == "offline":
            self.llm = OfflineLLM(model_mode=offline_model_mode)
        else:
            self.llm = OnlineLLM(provider)

    def generate(self, prompt, system_prompt=None):
        return self.llm.generate(prompt, system_prompt)
