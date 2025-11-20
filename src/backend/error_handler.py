"""
Error Handling and Logging Module for Disaster Literacy RAG System
Implements robust error detection, fallback responses, and logging
References: Lines 49-73 (Error Handling specification)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger

from config import (
    ERROR_LOG_FILE,
    GENERAL_LOG_FILE,
    LOG_LEVEL,
    ENABLE_FALLBACK,
    FALLBACK_MESSAGE
)


class DisasterRAGError(Exception):
    """Base exception for Disaster RAG system"""
    pass


class RetrievalError(DisasterRAGError):
    """Raised when KB retrieval fails - References: Line 57"""
    pass


class LLMInferenceError(DisasterRAGError):
    """Raised when LLM inference fails - References: Line 58"""
    pass


class PostProcessingError(DisasterRAGError):
    """Raised when post-processing fails - References: Line 59"""
    pass


class ErrorHandler:
    """
    Central error handling and logging system
    References: Lines 49-73 for error detection, fallback, logging, and user notification
    """
    
    def __init__(self):
        self._setup_logging()
        
    def _setup_logging(self):
        """
        Configure logging with file and console outputs
        References: Lines 67-69 (Logging with timestamps and query details)
        """
        # Remove default logger
        logger.remove()
        
        # Add console logging
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
        # Add general system logging
        logger.add(
            GENERAL_LOG_FILE,
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )
        
        # Add error-specific logging - References: Line 68
        logger.add(
            ERROR_LOG_FILE,
            level="ERROR",
            rotation="10 MB",
            retention="90 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            backtrace=True,
            diagnose=True
        )
        
        self.logger = logger
        
    def log_error(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        error_type: str = "general"
    ) -> None:
        """
        Log error with context for admin review
        References: Lines 67-69 (Record errors with timestamps and query details)
        
        Args:
            error: The exception that occurred
            context: Dictionary with query details, mode, etc.
            error_type: Type of error (retrieval, llm, post_processing)
        """
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context
        }
        
        self.logger.error(
            f"Error occurred: {error_type} | Query: {context.get('query', 'N/A')} | "
            f"Mode: {context.get('mode', 'N/A')} | Error: {str(error)}"
        )
        
    def generate_fallback_response(
        self, 
        error: Exception,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        mode: str = "Advisory"
    ) -> Dict[str, Any]:
        """
        Generate fallback response when errors occur
        References: Lines 60-66 (Fallback responses with simplified advisory)
        
        Args:
            error: The exception that triggered fallback
            retrieved_chunks: Available KB chunks if retrieval succeeded
            mode: Operation mode (Advisory/Educational/Simulation)
            
        Returns:
            Fallback response dictionary
        """
        if not ENABLE_FALLBACK:
            raise error
            
        fallback_response = {
            "success": False,
            "error_occurred": True,
            "error_type": type(error).__name__,
            "message": FALLBACK_MESSAGE,
            "content": self._create_fallback_content(retrieved_chunks, mode),
            "suggestions": self._generate_suggestions(mode)
        }
        
        return fallback_response
        
    def _create_fallback_content(
        self, 
        chunks: Optional[List[Dict[str, Any]]],
        mode: str
    ) -> str:
        """
        Create simplified advisory content from available chunks
        References: Lines 61-62 (Simplified advisory mode with actionable checklist)
        """
        if not chunks or len(chunks) == 0:
            return self._get_generic_safety_checklist(mode)
            
        # Extract key safety points from chunks
        content_lines = ["**Essential Safety Actions:**\n"]
        
        for i, chunk in enumerate(chunks[:3], 1):  # Use top 3 chunks
            text = chunk.get("text", "")
            source_id = chunk.get("chunk_id", f"source_{i}")
            
            # Extract first few sentences as safety tips
            sentences = text.split(".")[:2]
            if sentences:
                safety_tip = ". ".join(sentences).strip()
                content_lines.append(f"{i}. {safety_tip} [_{source_id}_]")
                
        return "\n".join(content_lines)
        
    def _get_generic_safety_checklist(self, mode: str) -> str:
        """
        Provide generic safety checklist when no KB chunks available
        """
        generic_checklist = """
**Essential Safety Actions:**

1. Stay calm and assess your immediate surroundings
2. Follow official emergency alerts and warnings
3. Evacuate if instructed by local authorities
4. Keep emergency supplies ready (water, food, first-aid kit)
5. Stay informed through official channels (radio, TV, emergency apps)
6. Contact emergency services (911 or local emergency number) if needed

**Note:** This is generic guidance. Please consult local authorities for specific instructions.
"""
        return generic_checklist.strip()
        
    def _generate_suggestions(self, mode: str) -> List[str]:
        """
        Generate alternative action suggestions for user
        References: Lines 70-72 (User notification and alternative actions)
        """
        suggestions = [
            "Contact local emergency services or authorities",
            "Check official government disaster websites",
            "Listen to local radio or TV emergency broadcasts",
            "Try rephrasing your query with different terms"
        ]
        
        if mode == "Simulation":
            suggestions.append("Switch to Advisory mode for immediate guidance")
            
        return suggestions
        
    def format_user_error_message(
        self, 
        error: Exception,
        mode: str
    ) -> str:
        """
        Format error message for user display without technical jargon
        References: Lines 70-72 (Clear communication without technical jargon)
        """
        # Map technical errors to user-friendly messages
        error_messages = {
            "RetrievalError": "Unable to find relevant information in the knowledge base.",
            "LLMInferenceError": "Unable to generate response at this time.",
            "PostProcessingError": "Unable to format the response properly.",
            "ConnectionError": "Unable to connect to online services.",
            "TimeoutError": "Request took too long. Please try again."
        }
        
        error_type = type(error).__name__
        user_message = error_messages.get(error_type, "An unexpected error occurred.")
        
        return f"⚠️ {user_message} Please try again or contact local authorities for assistance."
        
    def log_query(
        self, 
        query: str, 
        mode: str, 
        online: bool,
        user_id: Optional[str] = None
    ) -> None:
        """
        Log user query for monitoring and analytics
        """
        self.logger.info(
            f"Query received | Mode: {mode} | Online: {online} | "
            f"Query: {query[:100]}... | User: {user_id or 'anonymous'}"
        )
        
    def log_response(
        self, 
        query: str,
        response_time: float,
        success: bool,
        chunks_retrieved: int = 0
    ) -> None:
        """
        Log response metrics for evaluation
        References: Line 69 (Enable admins to review system performance)
        """
        self.logger.info(
            f"Response generated | Success: {success} | "
            f"Time: {response_time:.2f}s | Chunks: {chunks_retrieved} | "
            f"Query: {query[:50]}..."
        )


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(func):
    """
    Decorator for automatic error handling with fallback
    References: Lines 60-66 (Activates fallback when errors occur)
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                "function": func.__name__,
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200]
            }
            error_handler.log_error(e, context, error_type=type(e).__name__)
            
            # Re-raise for caller to handle
            raise
            
    return wrapper
