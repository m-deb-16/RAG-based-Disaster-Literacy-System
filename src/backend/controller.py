"""
Orchestration Controller for Disaster Literacy RAG System
Coordinates retrieval, LLM calls, post-processing, and answer checking
"""

import time
from typing import Dict, Any, List, Optional
import importlib
import config

from config import (
    OFFLINE_TOP_K_RETRIEVAL,
    OFFLINE_ECONOMY_TOP_K_RETRIEVAL,
    OFFLINE_POWER_TOP_K_RETRIEVAL,
    ONLINE_TOP_K_RETRIEVAL,
    MIN_MCQ_COUNT
)
from vector_store import VectorStore
from llm_wrapper import LLMWrapper
from templates import PromptTemplates
from post_processor import PostProcessor
from error_handler import error_handler, RetrievalError, LLMInferenceError, PostProcessingError


class DisasterRAGController:
    def __init__(self, online_mode: bool = False, provider: str = "openrouter", offline_model_mode: str = "economy"):
        self.online_mode = online_mode
        self.provider = provider.lower()
        self.offline_model_mode = offline_model_mode.lower()
        self.vector_store = None
        self.llm = None
        self.templates = PromptTemplates()
        self.post_processor = PostProcessor()

        self._initialize_components()

    def _initialize_components(self):
        try:
            error_handler.logger.info("Initializing vector store...")
            self.vector_store = VectorStore()

            mode = "online" if self.online_mode else "offline"
            if self.online_mode:
                error_handler.logger.info(f"Initializing {mode} LLM (provider: {self.provider})...")
                self.llm = LLMWrapper(mode=mode, provider=self.provider)
            else:
                error_handler.logger.info(f"Initializing {mode} LLM (model mode: {self.offline_model_mode})...")
                self.llm = LLMWrapper(mode=mode, provider=self.provider, offline_model_mode=self.offline_model_mode)

            error_handler.logger.info("Controller initialized successfully")

        except Exception as e:
            error_handler.logger.error(f"Failed to initialize controller: {e}")
            raise

    def _google_translate(self, text: str, target_lang: str, task="translate") -> str:
        """Uses Gemini only when provider=google for language work"""
        
        # Map language codes to full names for better translation
        lang_map = {
            'en': 'English',
            'bn': 'Bengali',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }

        try:
            from google.generativeai import GenerativeModel
            gm = GenerativeModel("gemini-2.5-flash")

            if task == "detect":
                prompt = f"""Detect the language of the following text and return ONLY the ISO 639-1 language code.
Valid codes: en (English), bn (Bengali), hi (Hindi), ta (Tamil), te (Telugu), mr (Marathi), gu (Gujarati), kn (Kannada), ml (Malayalam), pa (Punjabi), ur (Urdu)

Text: {text}

Return only the 2-letter code, nothing else."""
            else:
                # Convert language code to full name
                target_language = lang_map.get(target_lang, target_lang)
                prompt = f"""Translate the following text to {target_language}. Return ONLY the translation, no explanations or additional text.

Text to translate:
{text}

Translation:"""

            resp = gm.generate_content(prompt)
            result = resp.text.strip()
            
            # Clean up language detection result
            if task == "detect":
                # Extract just the language code if model added extra text
                import re
                match = re.search(r'\b(en|bn|hi|ta|te|mr|gu|kn|ml|pa|ur)\b', result.lower())
                if match:
                    result = match.group(1)
                    error_handler.logger.info(f"Detected language: {result} ({lang_map.get(result, 'Unknown')})")
            else:
                error_handler.logger.info(f"Translated to {target_lang}: {result[:50]}...")
            
            return result

        except Exception as e:
            error_handler.logger.warning(f"[Translation] Failed: {e}")
            return text
    #RAG begins
    def process_query(self, user_query: str, mode: str, disaster_type: Optional[str] = None):
        start_time = time.time()

        original_query = user_query
        user_lang = "en"
        
        # Store translation state before config reload (which might reset it)
        translation_enabled = config.ENABLE_TRANSLATION

        # ✅ Translation only when Gemini is being used in online mode
        error_handler.logger.info(f"Translation enabled: {translation_enabled}, Online: {self.online_mode}, Provider: {self.provider}, Non-ASCII: {not user_query.isascii()}")
        if translation_enabled and self.online_mode and self.provider == "google" and not user_query.isascii():
            error_handler.logger.info(f"Starting translation pipeline for query: {original_query[:50]}...")
            lang = self._google_translate(user_query, None, task="detect")
            user_lang = lang if lang else "en"
            error_handler.logger.info(f"Detected user language: {user_lang}")

            if user_lang != "en":
                error_handler.logger.info(f"Translating query from {user_lang} to English...")
                user_query = self._google_translate(user_query, "en")
                error_handler.logger.info(f"Translated query: {user_query[:50]}...")

        # Reload config
        importlib.reload(config)
        from config import (
            OFFLINE_TOP_K_RETRIEVAL,
            OFFLINE_ECONOMY_TOP_K_RETRIEVAL,
            OFFLINE_POWER_TOP_K_RETRIEVAL,
            ONLINE_TOP_K_RETRIEVAL
        )

        error_handler.log_query(user_query, mode, self.online_mode)

        if mode == "Simulation" and not self.online_mode:
            return {
                "success": False,
                "mode": mode,
                "query": original_query,
                "message": "Simulation mode requires online LLM.",
                "error_occurred": True
            }

        try:
            retrieved_chunks = self._retrieve_chunks(user_query, disaster_type)
            prompt, system_prompt = self._build_prompt(user_query, mode, retrieved_chunks, disaster_type)
            llm_output = self._call_llm(prompt, system_prompt)
            processed_response = self._post_process(llm_output, mode)

            # ✅ Translate output back to user language (only Gemini)
            if translation_enabled and self.online_mode and self.provider == "google" and user_lang != "en":
                error_handler.logger.info(f"Translating response back to {user_lang}...")
                content = processed_response.get("content", "")
                error_handler.logger.info(f"Original content length: {len(content)} chars")
                translated = self._google_translate(content, user_lang)
                error_handler.logger.info(f"Translated content length: {len(translated)} chars")
                processed_response["content"] = translated

            response_time = time.time() - start_time

            error_handler.log_response(original_query, response_time, True, len(retrieved_chunks))

            return {
                "success": True,
                "mode": mode,
                "query": original_query,
                "response": processed_response,
                "retrieved_chunks": retrieved_chunks,
                "response_time": response_time,
                "online_mode": self.online_mode
            }

        except (RetrievalError, LLMInferenceError, PostProcessingError) as e:
            error_handler.log_error(e, {"query": original_query, "mode": mode}, type(e).__name__)
            fallback = error_handler.generate_fallback_response(
                e, retrieved_chunks if "retrieved_chunks" in locals() else None, mode
            )
            return fallback

        except Exception as e:
            error_handler.log_error(e, {"query": original_query, "mode": mode}, "UnexpectedError")
            return {
                "success": False,
                "error_occurred": True,
                "message": "Unexpected error, try again.",
                "error_details": str(e)
            }


#look for relevant chunks in the vector
    def _retrieve_chunks(self, query, disaster_type):
        try:
            filter_metadata = {"disaster_type": disaster_type} if disaster_type else None
            
            # Determine top_k based on mode
            if self.online_mode:
                top_k = ONLINE_TOP_K_RETRIEVAL
            else:
                # Use model-specific retrieval count for offline mode
                if self.offline_model_mode == "power":
                    top_k = OFFLINE_POWER_TOP_K_RETRIEVAL
                elif self.offline_model_mode == "economy":
                    top_k = OFFLINE_ECONOMY_TOP_K_RETRIEVAL
                else:
                    top_k = OFFLINE_TOP_K_RETRIEVAL

            
            chunks = self.vector_store.search(query=query, top_k=top_k, filter_metadata=filter_metadata)

            if not chunks:
                raise RetrievalError("No relevant documents found")

            error_handler.logger.info(f"Retrieved {len(chunks)} chunks (mode: {self.offline_model_mode if not self.online_mode else 'online'})")
            return chunks

        except Exception as e:
            error_handler.logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Failed to retrieve chunks: {e}")

#Retrieves mode-specific system prompt from templates
 
    def _build_prompt(self, user_query, mode, retrieved_chunks, disaster_name):
        try:
            system_prompt = self.templates.get_system_prompt(mode)

            if mode == "Advisory":
                prompt = self.templates.advisory_template(user_query, retrieved_chunks)
            elif mode == "Educational":
                prompt = self.templates.educational_template(user_query, retrieved_chunks, disaster_name or "this disaster")
            elif mode == "Simulation":
                prompt = self.templates.simulation_template(user_query, retrieved_chunks, disaster_name or "scenario", MIN_MCQ_COUNT)
            else:
                raise ValueError("Invalid mode")

            return prompt, system_prompt

        except Exception as e:
            raise PostProcessingError(f"Failed to build prompt: {e}")
#LLM Generation
    def _call_llm(self, prompt, system_prompt):
        try:
            output = self.llm.generate(prompt, system_prompt)
            if not output.strip():
                raise LLMInferenceError("Empty response from LLM")
            return output
        except Exception as e:
            raise LLMInferenceError(f"LLM generation failed: {e}")

    def _post_process(self, llm_output, mode):
        try:
            if mode == "Simulation":
                return self.post_processor.parse_simulation_response(llm_output)
            elif mode == "Advisory":
                return self.post_processor.parse_advisory_response(llm_output)
            elif mode == "Educational":
                return self.post_processor.parse_educational_response(llm_output)
            return {"content": llm_output}
        except Exception as e:
            raise PostProcessingError(f"Failed to post-process: {e}")

    def check_simulation_answers(self, user_answers, mcqs):
        return self.post_processor.calculate_score(user_answers, mcqs)

    def get_system_stats(self):
        return {
            "online_mode": self.online_mode,
            "vector_store_stats": self.vector_store.get_stats() if self.vector_store else {}
        }
