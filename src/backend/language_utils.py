from google.generativeai import GenerativeModel
import re

class LanguageProcessor:
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = GenerativeModel(model_name)

    def detect_language(self, text: str) -> str:
        prompt = f"""
Detect the language of the following text. 
Return only the ISO language code (like 'en', 'bn', 'hi', 'ta'):

{text}
"""
        response = self.model.generate_content(prompt)
        lang = response.text.strip().lower()
        return re.sub(r'[^a-z]', '', lang)  # sanitize

    def translate(self, text: str, target_lang: str) -> str:
        prompt = f"""
Translate the following text to {target_lang}. Maintain meaning and tone:

{text}
"""
        response = self.model.generate_content(prompt)
        return response.text.strip()
