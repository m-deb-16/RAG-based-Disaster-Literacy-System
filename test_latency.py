"""
Latency Testing Script for Disaster Literacy RAG System
Tests response times for 2 offline models (economy, power) and 2 online providers (Google, OpenRouter)
"""

import sys
from pathlib import Path
import time
import json
import statistics
from datetime import datetime

# Add src/backend to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))

from controller import DisasterRAGController
import config

# Test queries covering different disaster types and complexities
TEST_QUERIES_OFFLINE = [
    {
        "query": "What should I do during a tsunami?",
        "mode": "Advisory",
        "disaster_type": "tsunami"
    },
    {
        "query": "How to prepare for earthquake?",
        "mode": "Educational",
        "disaster_type": "earthquake"
    }
]

TEST_QUERIES_ONLINE = [
    {
        "query": "What should I do during a tsunami?",
        "mode": "Advisory",
        "disaster_type": "tsunami"
    },
    {
        "query": "How to prepare for earthquake?",
        "mode": "Educational",
        "disaster_type": "earthquake"
    },
    {
        "query": "What are flood safety measures?",
        "mode": "Advisory",
        "disaster_type": "flood"
    },
    {
        "query": "Explain cyclone preparedness",
        "mode": "Educational",
        "disaster_type": "cyclone"
    },
    {
        "query": "What to do during a fire emergency?",
        "mode": "Advisory",
        "disaster_type": "fire"
    }
]


class LatencyTester:
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
    
    def test_offline_model(self, model_mode: str, num_runs: int = 3):
        """Test an offline model mode (economy or power)"""
        print(f"\n{'='*80}")
        print(f"TESTING OFFLINE MODE: {model_mode.upper()}")
        print(f"{'='*80}")
        
        try:
            print(f"Initializing controller for {model_mode} mode...")
            controller = DisasterRAGController(
                online_mode=False,
                offline_model_mode=model_mode
            )
            print("✅ Controller initialized\n")
            
            for idx, test_case in enumerate(TEST_QUERIES_OFFLINE, 1):
                query_latencies = []
                retrieval_times = []
                
                print(f"Query {idx}/{len(TEST_QUERIES_OFFLINE)}: {test_case['query'][:50]}...")
                
                for run in range(num_runs):
                    try:
                        start_time = time.time()
                        
                        result = controller.process_query(
                            user_query=test_case['query'],
                            mode=test_case['mode'],
                            disaster_type=test_case['disaster_type']
                        )
                        
                        end_time = time.time()
                        latency = end_time - start_time
                        query_latencies.append(latency)
                        
                        # Track retrieval info
                        if result.get('success'):
                            chunks_retrieved = len(result.get('retrieved_chunks', []))
                            retrieval_times.append({
                                'chunks': chunks_retrieved,
                                'time': latency
                            })
                        
                        print(f"  Run {run+1}: {latency:.2f}s - {'✅' if result.get('success') else '❌'}")
                        
                        # Small delay between runs
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  Run {run+1}: ❌ Error - {str(e)[:100]}")
                        continue
                
                if query_latencies:
                    avg_latency = statistics.mean(query_latencies)
                    self.results["offline"][model_mode].append(avg_latency)
                    
                    # Store detailed result
                    self.detailed_results.append({
                        "mode": "offline",
                        "model": model_mode,
                        "query": test_case['query'],
                        "disaster_type": test_case['disaster_type'],
                        "avg_latency": avg_latency,
                        "min_latency": min(query_latencies),
                        "max_latency": max(query_latencies),
                        "std_dev": statistics.stdev(query_latencies) if len(query_latencies) > 1 else 0,
                        "runs": len(query_latencies),
                        "retrieval_info": retrieval_times
                    })
                    
                    print(f"  Average: {avg_latency:.2f}s (min: {min(query_latencies):.2f}s, max: {max(query_latencies):.2f}s)")
                else:
                    print(f"  ⚠️ All runs failed")
                
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to test {model_mode} mode: {e}")
            return False
    
    def test_online_provider(self, provider: str, num_runs: int = 3):
        """Test an online provider (google or openrouter)"""
        print(f"\n{'='*80}")
        print(f"TESTING ONLINE PROVIDER: {provider.upper()}")
        print(f"{'='*80}")
        
        try:
            print(f"Initializing controller for {provider} provider...")
            controller = DisasterRAGController(
                online_mode=True,
                provider=provider
            )
            print("✅ Controller initialized\n")
            
            for idx, test_case in enumerate(TEST_QUERIES_ONLINE, 1):
                query_latencies = []
                retrieval_times = []
                
                print(f"Query {idx}/{len(TEST_QUERIES_ONLINE)}: {test_case['query'][:50]}...")
                
                for run in range(num_runs):
                    try:
                        start_time = time.time()
                        
                        result = controller.process_query(
                            user_query=test_case['query'],
                            mode=test_case['mode'],
                            disaster_type=test_case['disaster_type']
                        )
                        
                        end_time = time.time()
                        latency = end_time - start_time
                        query_latencies.append(latency)
                        
                        # Track retrieval info
                        if result.get('success'):
                            chunks_retrieved = len(result.get('retrieved_chunks', []))
                            retrieval_times.append({
                                'chunks': chunks_retrieved,
                                'time': latency
                            })
                        
                        print(f"  Run {run+1}: {latency:.2f}s - {'✅' if result.get('success') else '❌'}")
                        
                        # Delay to avoid rate limits
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"  Run {run+1}: ❌ Error - {str(e)[:100]}")
                        # Wait longer on error (might be rate limit)
                        time.sleep(5)
                        continue
                
                if query_latencies:
                    avg_latency = statistics.mean(query_latencies)
                    self.results["online"][provider].append(avg_latency)
                    
                    # Store detailed result
                    self.detailed_results.append({
                        "mode": "online",
                        "provider": provider,
                        "query": test_case['query'],
                        "disaster_type": test_case['disaster_type'],
                        "avg_latency": avg_latency,
                        "min_latency": min(query_latencies),
                        "max_latency": max(query_latencies),
                        "std_dev": statistics.stdev(query_latencies) if len(query_latencies) > 1 else 0,
                        "runs": len(query_latencies),
                        "retrieval_info": retrieval_times
                    })
                    
                    print(f"  Average: {avg_latency:.2f}s (min: {min(query_latencies):.2f}s, max: {max(query_latencies):.2f}s)")
                else:
                    print(f"  ⚠️ All runs failed")
                
                print()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to test {provider} provider: {e}")
            return False
    
    def generate_summary(self):
        """Generate summary statistics"""
        print(f"\n{'='*80}")
        print("LATENCY TEST SUMMARY")
        print(f"{'='*80}\n")
        
        # Offline results
        print("OFFLINE MODE:")
        print("-" * 80)
        for model_mode in ["economy", "power"]:
            latencies = self.results["offline"][model_mode]
            if latencies:
                print(f"\n{model_mode.upper()} Mode:")
                print(f"  Average Latency:    {statistics.mean(latencies):.2f}s")
                print(f"  Median Latency:     {statistics.median(latencies):.2f}s")
                print(f"  Min Latency:        {min(latencies):.2f}s")
                print(f"  Max Latency:        {max(latencies):.2f}s")
                print(f"  Std Deviation:      {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}s")
                print(f"  Total Queries:      {len(latencies)}")
            else:
                print(f"\n{model_mode.upper()} Mode: No successful queries")
        
        # Online results
        print("\n\nONLINE MODE:")
        print("-" * 80)
        for provider in ["google", "openrouter"]:
            latencies = self.results["online"][provider]
            if latencies:
                print(f"\n{provider.upper()} Provider:")
                print(f"  Average Latency:    {statistics.mean(latencies):.2f}s")
                print(f"  Median Latency:     {statistics.median(latencies):.2f}s")
                print(f"  Min Latency:        {min(latencies):.2f}s")
                print(f"  Max Latency:        {max(latencies):.2f}s")
                print(f"  Std Deviation:      {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}s")
                print(f"  Total Queries:      {len(latencies)}")
            else:
                print(f"\n{provider.upper()} Provider: No successful queries")
        
        # Comparison
        print("\n\nCOMPARATIVE ANALYSIS:")
        print("-" * 80)
        
        all_offline = []
        all_online = []
        
        for model_mode in ["economy", "power"]:
            all_offline.extend(self.results["offline"][model_mode])
        
        for provider in ["google", "openrouter"]:
            all_online.extend(self.results["online"][provider])
        
        if all_offline and all_online:
            print(f"\nOffline Average:   {statistics.mean(all_offline):.2f}s")
            print(f"Online Average:    {statistics.mean(all_online):.2f}s")
            print(f"Speed Ratio:       {statistics.mean(all_online) / statistics.mean(all_offline):.2f}x")
            print(f"                   (Online is {'faster' if statistics.mean(all_online) < statistics.mean(all_offline) else 'slower'})")
        
        # Save results to JSON
        self.save_results()
    
    def save_results(self):
        """Save results to JSON file"""
        output_file = f"latency_test_results_{self.timestamp}.json"
        
        results_data = {
            "timestamp": self.timestamp,
            "test_queries_offline": TEST_QUERIES_OFFLINE,
            "test_queries_online": TEST_QUERIES_ONLINE,
            "summary": self.results,
            "detailed_results": self.detailed_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {output_file}")


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("DISASTER RAG SYSTEM - LATENCY TESTING")
    print("="*80)
    print("\nThis script will test latency for:")
    print("  - Offline: Economy mode (Llama-2-7B) - 2 queries")
    print("  - Offline: Power mode (Qwen2-7B) - 2 queries")
    print("  - Online: Google Gemini - 5 queries")
    print("  - Online: OpenRouter (Qwen) - 5 queries")
    print(f"\nEach query will be run 3 times to get average latency")
    print(f"Offline queries: {len(TEST_QUERIES_OFFLINE)}, Online queries: {len(TEST_QUERIES_ONLINE)}")
    print("="*80)
    
    input("\nPress Enter to start testing...")
    
    tester = LatencyTester()
    
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
