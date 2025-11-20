"""
Post-Processing and MCQ Parser Module for Disaster Literacy RAG System
Parses LLM outputs, extracts MCQs, checks answers, validates evidence
References: Lines 90-96, 182-187 (Post-processing and MCQ handling)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from error_handler import error_handler, PostProcessingError


class PostProcessor:
    """
    Handles post-processing of LLM outputs
    References: Lines 90-96 (Post-processing specifications)
    """
    
    @staticmethod
    def parse_simulation_response(llm_output: str) -> Dict[str, Any]:
        """
        Parse simulation mode LLM output into structured format
        References: Lines 182-187 (MCQ generation structure)
        
        Expected format:
        SCENARIO: ...
        QUESTIONS:
        1) Q...
        A) ...
        B) ...
        C) ...
        D) ...
        ANSWER: X
        JUSTIFICATION: ...
        """
        try:
            result = {
                "scenario": "",
                "questions": [],
                "raw_output": llm_output
            }
            
            # Extract scenario
            scenario_match = re.search(
                r'SCENARIO:?\s*(.*?)(?=QUESTIONS:|$)',
                llm_output,
                re.DOTALL | re.IGNORECASE
            )
            if scenario_match:
                result["scenario"] = scenario_match.group(1).strip()
            
            # Extract questions
            questions_section = re.search(
                r'QUESTIONS:?\s*(.*)',
                llm_output,
                re.DOTALL | re.IGNORECASE
            )
            
            if questions_section:
                questions_text = questions_section.group(1)
                result["questions"] = PostProcessor._parse_mcqs(questions_text)
            
            if not result["questions"]:
                raise PostProcessingError("No MCQs found in LLM output")
                
            return result
            
        except Exception as e:
            error_handler.logger.error(f"Failed to parse simulation response: {e}")
            raise PostProcessingError(f"MCQ parsing failed: {e}")
    
    @staticmethod
    def _parse_mcqs(text: str) -> List[Dict[str, Any]]:
        """
        Parse individual MCQs from text
        """
        mcqs = []
        
        # Pattern to match question blocks
        # Matches: 1) Question... A) ... B) ... C) ... D) ... ANSWER: X JUSTIFICATION: ...
        question_pattern = r'(\d+)\)\s*(.*?)(?=\d+\)|$)'
        question_blocks = re.findall(question_pattern, text, re.DOTALL)
        
        for q_num, q_block in question_blocks:
            try:
                mcq = PostProcessor._parse_single_mcq(q_num, q_block)
                if mcq:
                    mcqs.append(mcq)
            except Exception as e:
                error_handler.logger.warning(f"Failed to parse question {q_num}: {e}")
                continue
        
        return mcqs
    
    @staticmethod
    def _parse_single_mcq(q_num: str, q_block: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single MCQ from text block
        """
        # Extract question text (before options)
        question_match = re.search(r'^(.*?)(?=[A-D]\))', q_block, re.DOTALL)
        if not question_match:
            return None
        question_text = question_match.group(1).strip()
        
        # Extract options
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            option_pattern = rf'{letter}\)\s*(.*?)(?=[A-D]\)|ANSWER:|$)'
            option_match = re.search(option_pattern, q_block, re.DOTALL | re.IGNORECASE)
            if option_match:
                options[letter] = option_match.group(1).strip()
        
        if len(options) < 4:
            return None
        
        # Extract answer
        answer_match = re.search(
            r'ANSWER:\s*([A-D])',
            q_block,
            re.IGNORECASE
        )
        answer = answer_match.group(1).upper() if answer_match else None
        
        # Extract justification
        just_match = re.search(
            r'JUSTIFICATION:\s*(.*?)(?=\n\n|$)',
            q_block,
            re.DOTALL | re.IGNORECASE
        )
        justification = just_match.group(1).strip() if just_match else ""
        
        # Extract cited sources
        cited_sources = PostProcessor.extract_citations(justification)
        
        return {
            "question_num": int(q_num),
            "question": question_text,
            "options": options,
            "correct_answer": answer,
            "justification": justification,
            "cited_sources": cited_sources
        }
    
    @staticmethod
    def check_answer(
        user_answer: str,
        correct_answer: str
    ) -> Tuple[bool, str]:
        """
        Check if user answer matches correct answer
        References: Lines 94-96 (Answer checking)
        
        Returns:
            Tuple of (is_correct, feedback_message)
        """
        user_answer = user_answer.strip().upper()
        correct_answer = correct_answer.strip().upper()
        
        is_correct = user_answer == correct_answer
        
        if is_correct:
            feedback = f"✓ Correct! The answer is {correct_answer}."
        else:
            feedback = f"✗ Incorrect. You answered {user_answer}, but the correct answer is {correct_answer}."
        
        return is_correct, feedback
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """
        Extract source citations from text
        References: Line 93 (Evidence chunk IDs)
        
        Finds patterns like [source_id] or [chunk_123]
        """
        citations = re.findall(r'\[([\w_\-\.]+)\]', text)
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def parse_advisory_response(llm_output: str) -> Dict[str, Any]:
        """
        Parse advisory mode output
        References: Lines 90-93 (Extract bullet lists + evidence references)
        """
        result = {
            "content": llm_output,
            "cited_sources": PostProcessor.extract_citations(llm_output),
            "action_items": []
        }
        
        # Extract bullet points as action items, excluding citations
        # This regex handles various bullet types (-, *, •) and captures the full item before any [citation]
        bullets = re.findall(r'^[ \t]*[•\-\*]\s*(.+?)(?=\s*\[[\w_\-\.]+\]|\n[ \t]*[•\-\*]|\n\n|$)', llm_output, re.MULTILINE)
        result["action_items"] = [b.strip() for b in bullets]
        
        return result
    
    @staticmethod
    def parse_educational_response(llm_output: str) -> Dict[str, Any]:
        """
        Parse educational mode output
        """
        result = {
            "content": llm_output,
            "cited_sources": PostProcessor.extract_citations(llm_output)
        }
        
        # Try to extract overview, do's/don'ts, FAQ sections
        overview_match = re.search(
            r'(?:Overview|Introduction):?\s*(.*?)(?=Do\'s|Don\'ts|FAQ|$)',
            llm_output,
            re.DOTALL | re.IGNORECASE
        )
        if overview_match:
            result["overview"] = overview_match.group(1).strip()
        
        return result
    
    @staticmethod
    def calculate_score(user_answers: Dict[int, str], mcqs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate simulation test score
        References: Line 240 (Total score display)
        """
        total_questions = len(mcqs)
        correct_count = 0
        results = []
        
        for mcq in mcqs:
            q_num = mcq["question_num"]
            correct_answer = mcq["correct_answer"]
            user_answer = user_answers.get(q_num, "").upper()
            
            is_correct = user_answer == correct_answer
            if is_correct:
                correct_count += 1
            
            results.append({
                "question_num": q_num,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "question": mcq["question"],
                "justification": mcq["justification"]
            })
        
        score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "incorrect_answers": total_questions - correct_count,
            "score_percentage": score_percentage,
            "results": results,
            "pass_threshold": 70,  # 70% to pass
            "passed": score_percentage >= 70
        }
