"""
Prompt Templates Module for Disaster Literacy RAG System
Defines Advisory, Educational, and Simulation mode templates
References: Lines 108-181 (Template specifications)
"""

from typing import List, Dict, Any
from config import GROUNDING_INSTRUCTION


class PromptTemplates:
    """
    Manages prompt templates for different operation modes
    References: Lines 108-181
    """
    
    @staticmethod
    def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context for LLM
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_id = chunk.get('chunk_id', f'source_{i}')
            text = chunk.get('text', '')
            context_parts.append(f"[{source_id}]\n{text}\n")
        return "\n".join(context_parts)
    
    @staticmethod
    def advisory_template(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Advisory mode template
        References: Lines 110-134 (Advisory Template)
        
        Provides concise, actionable disaster safety advice
        """
        context = PromptTemplates.format_context(retrieved_chunks)
        
        template = f"""CONTEXT:

{context}

INSTRUCTIONS TO MODEL:

You are a concise disaster safety advisor. Use ONLY the information from CONTEXT (do not hallucinate). Cite the source id(s) used in square brackets after each recommendation. Provide:
 
1) Short (3-6 bullet) immediate action checklist.
 
2) Safety do's and don'ts.
 
3) Any local resources/contact numbers found (if any).
 
Keep answer concise and practical. Format bullets and include [source_id] tags.

{GROUNDING_INSTRUCTION}

USER QUERY:

{user_query}

RESPONSE:"""
        
        return template
    
    @staticmethod
    def educational_template(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        disaster_name: str = "the disaster"
    ) -> str:
        """
        Educational mode template
        References: Lines 136-148 (Educational Template)
        
        Teaches users about disaster preparedness and response
        """
        context = PromptTemplates.format_context(retrieved_chunks)
        
        template = f"""CONTEXT:

{context}

INSTRUCTIONS:

Teach the user about {disaster_name}. Start with a one-paragraph overview, then clear do's and don'ts, then a short FAQ (3 Qs). Use only CONTEXT and cite sources. Aim for readability for rural users (simple language).

{GROUNDING_INSTRUCTION}

USER QUERY:

{user_query}

RESPONSE:"""
        
        return template
    
    @staticmethod
    def simulation_template(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        disaster_name: str = "disaster scenario",
        min_mcqs: int = 10
    ) -> str:
        """
        Simulation mode template for MCQ generation
        References: Lines 150-180 (Simulation Template)
        
        Generates realistic scenarios with multiple-choice questions
        """
        context = PromptTemplates.format_context(retrieved_chunks)
        
        template = f"""CONTEXT:

{context}

INSTRUCTIONS:

Generate a realistic scenario related to {disaster_name} in 2-3 sentences. Then generate at least {min_mcqs} multiple-choice questions (4 options each), each with one correct option. For each question, provide a one-sentence explanation that cites the supporting CONTEXT source(s). Format as:

SCENARIO:

...

QUESTIONS:

1) Q...

A) ...

B) ...

C) ...

D) ...

ANSWER: <letter>

JUSTIFICATION: <one line with [source_id]>

2) Q...

...

{GROUNDING_INSTRUCTION}

USER QUERY:

{user_query}

RESPONSE:"""
        
        return template
    
    @staticmethod
    def answer_verification_template(
        question: str,
        user_answer: str,
        canonical_answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Template for verifying user answers in simulation mode
        References: Lines 94-96 (Answer checking with KB support)
        """
        context = PromptTemplates.format_context(retrieved_chunks)
        
        template = f"""CONTEXT:

{context}

TASK:

The user answered a disaster preparedness question. Verify if their answer is correct and provide a brief explanation using the CONTEXT.

QUESTION: {question}

USER'S ANSWER: {user_answer}

CORRECT ANSWER: {canonical_answer}

Provide:
1) Is the user correct? (Yes/No)
2) Brief explanation (1-2 sentences) citing relevant [source_id]

RESPONSE:"""
        
        return template
    
    @staticmethod
    def get_system_prompt(mode: str) -> str:
        """
        Get system prompt for different modes
        References: Lines 82, 88, 191-196 (System instructions for grounding)
        """
        base_prompt = (
            "You are a disaster literacy assistant helping people prepare for and respond to emergencies. "
            f"You are operating in {mode} mode. "
            f"{GROUNDING_INSTRUCTION} "
            "Always prioritize safety and accuracy. If uncertain, direct users to contact local authorities."
        )
        
        mode_specific = {
            "Advisory": " Provide concise, actionable safety guidance.",
            "Educational": " Teach users about disaster preparedness in simple, accessible language.",
            "Simulation": " Create realistic scenarios and test user knowledge with multiple-choice questions."
        }
        
        return base_prompt + mode_specific.get(mode, "")
