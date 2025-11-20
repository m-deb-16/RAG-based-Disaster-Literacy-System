"""
Accuracy Testing Script for Disaster Literacy RAG System
Tests response quality and accuracy for offline and online models
Metrics: Relevance, Completeness, Citation Accuracy, Coherence, Actionability
"""

import sys
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Add src/backend to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))

from controller import DisasterRAGController
import config

# Test cases with ground truth and evaluation criteria
TEST_CASES_OFFLINE = [
    {
        "query": "What should I do during a tsunami?",
        "mode": "Advisory",
        "disaster_type": "tsunami",
        "expected_keywords": ["evacuate", "high ground", "warning", "coast", "inland"],
        "expected_actions": ["move to higher ground", "evacuate", "stay away from coast"],
        "evaluation_criteria": {
            "must_mention_evacuation": True,
            "must_mention_high_ground": True,
            "should_be_actionable": True
        }
    },
    {
        "query": "How to prepare for earthquake?",
        "mode": "Educational",
        "disaster_type": "earthquake",
        "expected_keywords": ["drop", "cover", "hold", "emergency kit", "safe place"],
        "expected_actions": ["prepare emergency kit", "identify safe spots", "secure furniture"],
        "evaluation_criteria": {
            "must_mention_dropcoverhold": False,
            "must_mention_preparation": True,
            "should_be_informative": True
        }
    }
]

TEST_CASES_ONLINE = [
    {
        "query": "What should I do during a tsunami?",
        "mode": "Advisory",
        "disaster_type": "tsunami",
        "expected_keywords": ["evacuate", "high ground", "warning", "coast", "inland"],
        "expected_actions": ["move to higher ground", "evacuate", "stay away from coast"],
        "evaluation_criteria": {
            "must_mention_evacuation": True,
            "must_mention_high_ground": True,
            "should_be_actionable": True
        }
    },
    {
        "query": "How to prepare for earthquake?",
        "mode": "Educational",
        "disaster_type": "earthquake",
        "expected_keywords": ["drop", "cover", "hold", "emergency kit", "safe place"],
        "expected_actions": ["prepare emergency kit", "identify safe spots", "secure furniture"],
        "evaluation_criteria": {
            "must_mention_dropcoverhold": False,
            "must_mention_preparation": True,
            "should_be_informative": True
        }
    },
    {
        "query": "What are flood safety measures?",
        "mode": "Advisory",
        "disaster_type": "flood",
        "expected_keywords": ["water", "evacuation", "avoid", "flooded areas", "higher ground"],
        "expected_actions": ["evacuate if ordered", "avoid flooded areas", "move to higher ground"],
        "evaluation_criteria": {
            "must_mention_avoidance": True,
            "must_mention_evacuation": True,
            "should_be_actionable": True
        }
    },
    {
        "query": "Explain cyclone preparedness",
        "mode": "Educational",
        "disaster_type": "cyclone",
        "expected_keywords": ["shelter", "supplies", "warning", "evacuation", "wind"],
        "expected_actions": ["prepare emergency supplies", "identify shelter", "monitor warnings"],
        "evaluation_criteria": {
            "must_mention_shelter": True,
            "must_mention_supplies": True,
            "should_be_informative": True
        }
    },
    {
        "query": "What to do during a fire emergency?",
        "mode": "Advisory",
        "disaster_type": "fire",
        "expected_keywords": ["exit", "smoke", "crawl", "call", "evacuation"],
        "expected_actions": ["exit building", "stay low", "call emergency services"],
        "evaluation_criteria": {
            "must_mention_exit": True,
            "must_mention_smoke": True,
            "should_be_actionable": True
        }
    }
]


class AccuracyTester:
    def __init__(self):
        self.results = {
            "offline": {
                "economy": [],
                "power": []
            },
            "online": {
                "google": [],
                "openrouter": []
            }
        }
        self.detailed_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def evaluate_response(self, response: Dict[str, Any], test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate response quality using multiple metrics
        Returns scores and detailed analysis
        """
        if not response.get('success'):
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "citation_score": 0.0,
                "coherence_score": 0.0,
                "actionability_score": 0.0,
                "error": response.get('message', 'Unknown error')
            }
        
        response_data = response.get('response', {})
        content = response_data.get('content', '').lower()
        cited_sources = response_data.get('cited_sources', [])
        action_items = response_data.get('action_items', [])
        
        # 1. Relevance Score: Check for expected keywords
        relevance_score = self._calculate_relevance(content, test_case['expected_keywords'])
        
        # 2. Completeness Score: Check if key actions/concepts are covered
        completeness_score = self._calculate_completeness(content, test_case['expected_actions'])
        
        # 3. Citation Score: Evaluate quality and presence of citations
        citation_score = self._calculate_citation_score(cited_sources, content)
        
        # 4. Coherence Score: Evaluate structure and readability
        coherence_score = self._calculate_coherence(content)
        
        # 5. Actionability Score: Check if response provides actionable guidance (for Advisory mode)
        actionability_score = self._calculate_actionability(
            content, 
            action_items, 
            test_case['mode']
        )
        
        # 6. Criteria Compliance: Check specific evaluation criteria
        criteria_compliance = self._check_criteria_compliance(content, test_case['evaluation_criteria'])
        
        # Calculate overall score (weighted average)
        weights = {
            'relevance': 0.25,
            'completeness': 0.25,
            'citation': 0.15,
            'coherence': 0.15,
            'actionability': 0.20
        }
        
        overall_score = (
            weights['relevance'] * relevance_score +
            weights['completeness'] * completeness_score +
            weights['citation'] * citation_score +
            weights['coherence'] * coherence_score +
            weights['actionability'] * actionability_score
        )
        
        return {
            "overall_score": round(overall_score, 2),
            "relevance_score": round(relevance_score, 2),
            "completeness_score": round(completeness_score, 2),
            "citation_score": round(citation_score, 2),
            "coherence_score": round(coherence_score, 2),
            "actionability_score": round(actionability_score, 2),
            "criteria_compliance": criteria_compliance,
            "response_length": len(content),
            "num_citations": len(cited_sources),
            "num_action_items": len(action_items)
        }
    
    def _calculate_relevance(self, content: str, expected_keywords: List[str]) -> float:
        """Calculate relevance based on keyword presence"""
        if not expected_keywords:
            return 1.0
        
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in content)
        return min(matches / len(expected_keywords), 1.0)
    
    def _calculate_completeness(self, content: str, expected_actions: List[str]) -> float:
        """Calculate completeness based on action coverage"""
        if not expected_actions:
            return 1.0
        
        # Check if key concepts from expected actions are present
        matches = 0
        for action in expected_actions:
            # Extract key words from action (ignore common words)
            key_words = [w for w in action.lower().split() if len(w) > 3]
            if any(word in content for word in key_words):
                matches += 1
        
        return min(matches / len(expected_actions), 1.0)
    
    def _calculate_citation_score(self, cited_sources: List[str], content: str) -> float:
        """Evaluate citation quality"""
        if not cited_sources:
            # Penalize missing citations
            return 0.3
        
        # Check for citation markers in content
        citation_patterns = [r'\[.*?\]', r'\(.*?\)', r'source:', r'according to']
        has_citation_markers = any(re.search(pattern, content, re.IGNORECASE) for pattern in citation_patterns)
        
        # Score based on number of sources and markers
        num_sources = len(cited_sources)
        source_score = min(num_sources / 3, 1.0)  # Optimal is 3+ sources
        marker_score = 1.0 if has_citation_markers else 0.7
        
        return (source_score + marker_score) / 2
    
    def _calculate_coherence(self, content: str) -> float:
        """Evaluate structure and coherence"""
        if not content or len(content) < 50:
            return 0.3
        
        score = 0.0
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) >= 3:
            score += 0.4
        elif len(valid_sentences) >= 2:
            score += 0.2
        
        # Check for organization (bullet points, numbered lists, etc.)
        has_structure = bool(re.search(r'[\n\-\*\d+\.]', content))
        if has_structure:
            score += 0.3
        
        # Check for reasonable length (not too short or excessively long)
        if 100 <= len(content) <= 2000:
            score += 0.3
        elif 50 <= len(content) < 100 or 2000 < len(content) <= 3000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_actionability(self, content: str, action_items: List[str], mode: str) -> float:
        """Evaluate actionability (especially for Advisory mode)"""
        if mode != "Advisory":
            # Less critical for Educational mode
            return 0.8 if len(content) > 100 else 0.5
        
        score = 0.0
        
        # Check for action items
        if action_items and len(action_items) >= 3:
            score += 0.5
        elif action_items:
            score += 0.3
        
        # Check for imperative verbs (action words)
        action_verbs = ['do', 'go', 'move', 'evacuate', 'prepare', 'avoid', 'stay', 'call', 'contact']
        verb_count = sum(1 for verb in action_verbs if verb in content)
        score += min(verb_count / 5, 0.5)
        
        return min(score, 1.0)
    
    def _check_criteria_compliance(self, content: str, criteria: Dict[str, bool]) -> Dict[str, bool]:
        """Check if specific evaluation criteria are met"""
        compliance = {}
        
        for criterion, required in criteria.items():
            if 'evacuation' in criterion:
                compliance[criterion] = 'evacuat' in content
            elif 'high_ground' in criterion or 'higher_ground' in criterion:
                compliance[criterion] = 'high' in content and 'ground' in content
            elif 'dropcoverhold' in criterion:
                compliance[criterion] = ('drop' in content or 'cover' in content or 'hold' in content)
            elif 'preparation' in criterion or 'prepare' in criterion:
                compliance[criterion] = 'prepar' in content
            elif 'avoidance' in criterion or 'avoid' in criterion:
                compliance[criterion] = 'avoid' in content
            elif 'shelter' in criterion:
                compliance[criterion] = 'shelter' in content
            elif 'supplies' in criterion:
                compliance[criterion] = ('suppl' in content or 'kit' in content)
            elif 'exit' in criterion:
                compliance[criterion] = 'exit' in content
            elif 'smoke' in criterion:
                compliance[criterion] = 'smoke' in content
            elif 'actionable' in criterion:
                # Check for imperative language
                compliance[criterion] = any(word in content for word in ['should', 'must', 'do', 'move', 'go'])
            elif 'informative' in criterion:
                # Check for sufficient length and structure
                compliance[criterion] = len(content) > 150
            else:
                compliance[criterion] = True
        
        return compliance
    
    def test_model_configuration(self, mode_type: str, identifier: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a specific model configuration with a test case
        mode_type: 'offline' or 'online'
        identifier: 'economy'/'power' for offline, 'google'/'openrouter' for online
        """
        try:
            if mode_type == "offline":
                controller = DisasterRAGController(
                    online_mode=False,
                    offline_model_mode=identifier
                )
            else:
                controller = DisasterRAGController(
                    online_mode=True,
                    provider=identifier
                )
            
            # Process query
            result = controller.process_query(
                user_query=test_case['query'],
                mode=test_case['mode'],
                disaster_type=test_case['disaster_type']
            )
            
            # Evaluate response
            evaluation = self.evaluate_response(result, test_case)
            
            return {
                "success": True,
                "result": result,
                "evaluation": evaluation
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "evaluation": {
                    "overall_score": 0.0,
                    "error": str(e)
                }
            }
    
    def test_offline_model(self, model_mode: str):
        """Test an offline model mode"""
        print(f"\n{'='*80}")
        print(f"TESTING OFFLINE MODE: {model_mode.upper()}")
        print(f"{'='*80}\n")
        
        for idx, test_case in enumerate(TEST_CASES_OFFLINE, 1):
            print(f"Test {idx}/{len(TEST_CASES_OFFLINE)}: {test_case['query'][:60]}...")
            
            test_result = self.test_model_configuration("offline", model_mode, test_case)
            
            if test_result['success']:
                evaluation = test_result['evaluation']
                print(f"  Overall Score: {evaluation['overall_score']:.2f}/1.00")
                print(f"  - Relevance:      {evaluation['relevance_score']:.2f}")
                print(f"  - Completeness:   {evaluation['completeness_score']:.2f}")
                print(f"  - Citations:      {evaluation['citation_score']:.2f}")
                print(f"  - Coherence:      {evaluation['coherence_score']:.2f}")
                print(f"  - Actionability:  {evaluation['actionability_score']:.2f}")
                
                self.results["offline"][model_mode].append(evaluation['overall_score'])
                
                # Store detailed result
                self.detailed_results.append({
                    "mode": "offline",
                    "model": model_mode,
                    "query": test_case['query'],
                    "disaster_type": test_case['disaster_type'],
                    "evaluation": evaluation,
                    "response_snippet": test_result['result']['response']['content'][:200] if test_result['result'].get('success') else ""
                })
            else:
                print(f"  ❌ Error: {test_result['error'][:100]}")
            
            print()
    
    def test_online_provider(self, provider: str):
        """Test an online provider"""
        print(f"\n{'='*80}")
        print(f"TESTING ONLINE PROVIDER: {provider.upper()}")
        print(f"{'='*80}\n")
        
        for idx, test_case in enumerate(TEST_CASES_ONLINE, 1):
            print(f"Test {idx}/{len(TEST_CASES_ONLINE)}: {test_case['query'][:60]}...")
            
            test_result = self.test_model_configuration("online", provider, test_case)
            
            if test_result['success']:
                evaluation = test_result['evaluation']
                print(f"  Overall Score: {evaluation['overall_score']:.2f}/1.00")
                print(f"  - Relevance:      {evaluation['relevance_score']:.2f}")
                print(f"  - Completeness:   {evaluation['completeness_score']:.2f}")
                print(f"  - Citations:      {evaluation['citation_score']:.2f}")
                print(f"  - Coherence:      {evaluation['coherence_score']:.2f}")
                print(f"  - Actionability:  {evaluation['actionability_score']:.2f}")
                
                self.results["online"][provider].append(evaluation['overall_score'])
                
                # Store detailed result
                self.detailed_results.append({
                    "mode": "online",
                    "provider": provider,
                    "query": test_case['query'],
                    "disaster_type": test_case['disaster_type'],
                    "evaluation": evaluation,
                    "response_snippet": test_result['result']['response']['content'][:200] if test_result['result'].get('success') else ""
                })
            else:
                print(f"  ❌ Error: {test_result['error'][:100]}")
            
            print()
            
            # Delay for online providers to avoid rate limits
            import time
            time.sleep(2)
    
    def generate_summary(self):
        """Generate comprehensive summary of accuracy tests"""
        print(f"\n{'='*80}")
        print("ACCURACY TEST SUMMARY")
        print(f"{'='*80}\n")
        
        # Calculate averages
        print("OFFLINE MODE:")
        print("-" * 80)
        for model_mode in ["economy", "power"]:
            scores = self.results["offline"][model_mode]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"\n{model_mode.upper()} Mode:")
                print(f"  Average Score:      {avg_score:.2f}/1.00")
                print(f"  Min Score:          {min(scores):.2f}/1.00")
                print(f"  Max Score:          {max(scores):.2f}/1.00")
                print(f"  Success Rate:       {len(scores)}/{len(TEST_CASES_OFFLINE)} ({len(scores)/len(TEST_CASES_OFFLINE)*100:.0f}%)")
            else:
                print(f"\n{model_mode.upper()} Mode: No successful tests")
        
        print("\n\nONLINE MODE:")
        print("-" * 80)
        for provider in ["google", "openrouter"]:
            scores = self.results["online"][provider]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"\n{provider.upper()} Provider:")
                print(f"  Average Score:      {avg_score:.2f}/1.00")
                print(f"  Min Score:          {min(scores):.2f}/1.00")
                print(f"  Max Score:          {max(scores):.2f}/1.00")
                print(f"  Success Rate:       {len(scores)}/{len(TEST_CASES_ONLINE)} ({len(scores)/len(TEST_CASES_ONLINE)*100:.0f}%)")
            else:
                print(f"\n{provider.upper()} Provider: No successful tests")
        
        # Comparative analysis
        print("\n\nCOMPARATIVE ANALYSIS:")
        print("-" * 80)
        
        all_offline = []
        all_online = []
        
        for model_mode in ["economy", "power"]:
            all_offline.extend(self.results["offline"][model_mode])
        
        for provider in ["google", "openrouter"]:
            all_online.extend(self.results["online"][provider])
        
        if all_offline and all_online:
            offline_avg = sum(all_offline) / len(all_offline)
            online_avg = sum(all_online) / len(all_online)
            
            print(f"\nOffline Average:   {offline_avg:.2f}/1.00")
            print(f"Online Average:    {online_avg:.2f}/1.00")
            print(f"Difference:        {abs(online_avg - offline_avg):.2f}")
            print(f"Better Performer:  {'Online' if online_avg > offline_avg else 'Offline'}")
        
        # Grade assignment
        print("\n\nPERFORMANCE GRADES:")
        print("-" * 80)
        
        def get_grade(score):
            if score >= 0.9: return "A (Excellent)"
            elif score >= 0.8: return "B (Good)"
            elif score >= 0.7: return "C (Fair)"
            elif score >= 0.6: return "D (Poor)"
            else: return "F (Fail)"
        
        for model_mode in ["economy", "power"]:
            scores = self.results["offline"][model_mode]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"Offline {model_mode.capitalize()}: {get_grade(avg)}")
        
        for provider in ["google", "openrouter"]:
            scores = self.results["online"][provider]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"Online {provider.capitalize()}: {get_grade(avg)}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        output_file = f"accuracy_test_results_{self.timestamp}.json"
        
        results_data = {
            "timestamp": self.timestamp,
            "test_cases_offline": TEST_CASES_OFFLINE,
            "test_cases_online": TEST_CASES_ONLINE,
            "summary": self.results,
            "detailed_results": self.detailed_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {output_file}")


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("DISASTER RAG SYSTEM - ACCURACY TESTING")
    print("="*80)
    print("\nThis script will evaluate response accuracy for:")
    print("  - Offline: Economy mode (Llama-2-7B) - 2 test cases")
    print("  - Offline: Power mode (Qwen2-7B) - 2 test cases")
    print("  - Online: Google Gemini - 5 test cases")
    print("  - Online: OpenRouter (Qwen) - 5 test cases")
    print(f"\nMetrics evaluated:")
    print("  • Relevance: Keyword matching with expected terms")
    print("  • Completeness: Coverage of key actions/concepts")
    print("  • Citations: Quality and presence of source citations")
    print("  • Coherence: Structure and readability")
    print("  • Actionability: Practical guidance (for Advisory mode)")
    print(f"\nOffline test cases: {len(TEST_CASES_OFFLINE)}, Online test cases: {len(TEST_CASES_ONLINE)}")
    print("="*80)
    
    input("\nPress Enter to start testing...")
    
    tester = AccuracyTester()
    
    # Test offline models
    print("\n\n📍 PHASE 1: OFFLINE MODELS")
    print("="*80)
    tester.test_offline_model("economy")
    tester.test_offline_model("power")
    
    # Test online providers
    print("\n\n📍 PHASE 2: ONLINE PROVIDERS")
    print("="*80)
    print("\n⚠️ Note: Online tests include delays to avoid rate limits")
    tester.test_online_provider("google")
    tester.test_online_provider("openrouter")
    
    # Generate summary
    tester.generate_summary()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
