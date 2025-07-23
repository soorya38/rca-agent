#!/usr/bin/env python3
"""
High-Performance Load Testing Script for Helios
This script demonstrates processing thousands of logs per second
"""

import asyncio
import aiohttp
import time
import json
import random
from typing import List, Dict
import uvloop

# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Sample log patterns for testing
SAMPLE_LOGS = [
    """2024-01-15 10:30:15 ERROR [UserService] Database connection failed: Connection timeout
2024-01-15 10:30:15 WARN [ConnectionPool] Pool exhausted, max connections: 20
2024-01-15 10:30:16 ERROR [UserService] Failed to authenticate user: Database unavailable""",
    
    """2024-01-15 14:22:33 ERROR [PaymentService] Credit card validation failed
2024-01-15 14:22:33 INFO [PaymentService] Retrying payment with backup processor
2024-01-15 14:22:34 ERROR [PaymentService] Backup processor also failed
2024-01-15 14:22:35 CRITICAL [PaymentService] Payment pipeline completely down""",
    
    """2024-01-15 09:15:22 WARN [LoadBalancer] Server node-3 not responding
2024-01-15 09:15:23 ERROR [LoadBalancer] Failed to route request: no healthy backends
2024-01-15 09:15:24 INFO [LoadBalancer] Attempting failover to backup cluster
2024-01-15 09:15:25 ERROR [LoadBalancer] Failover failed: backup cluster unreachable""",
    
    """2024-01-15 16:45:12 ERROR [APIGateway] Rate limit exceeded for client 192.168.1.100
2024-01-15 16:45:13 WARN [APIGateway] Suspicious activity detected: too many requests
2024-01-15 16:45:14 INFO [APIGateway] Temporarily blocking client for 300 seconds
2024-01-15 16:45:15 ERROR [APIGateway] Still receiving requests from blocked client""",
    
    """2024-01-15 11:30:45 ERROR [CacheService] Redis connection lost
2024-01-15 11:30:46 WARN [CacheService] Falling back to in-memory cache
2024-01-15 11:30:47 ERROR [CacheService] Memory cache capacity exceeded
2024-01-15 11:30:48 CRITICAL [CacheService] No caching available, performance degraded"""
]

SAMPLE_QUESTIONS = [
    "What went wrong?",
    "What is the root cause of this issue?",
    "How can I fix this problem?",
    "What are the main issues here?",
    "Analyze this failure",
    "What should I do to resolve this?",
    "Identify the critical problems",
    "What caused this outage?"
]

class HeliosLoadTester:
    """High-performance load tester for Helios"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "response_times": []
        }
    
    async def initialize(self):
        """Initialize HTTP session with optimizations"""
        connector = aiohttp.TCPConnector(
            limit=1000,  # Maximum number of connections
            limit_per_host=100,  # Maximum connections per host
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    async def send_single_request(self, session_id: str = None) -> Dict:
        """Send a single analysis request"""
        logs = random.choice(SAMPLE_LOGS)
        question = random.choice(SAMPLE_QUESTIONS)
        
        payload = {
            "logs": logs,
            "question": question,
            "session_id": session_id or f"load_test_{random.randint(1000, 9999)}",
            "priority": random.randint(1, 10),
            "use_cache": True
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.base_url}/chat", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    self.results["successful_requests"] += 1
                    self.results["response_times"].append(processing_time)
                    self.results["min_response_time"] = min(self.results["min_response_time"], processing_time)
                    self.results["max_response_time"] = max(self.results["max_response_time"], processing_time)
                    
                    if result.get("cached", False):
                        self.results["cache_hits"] += 1
                    
                    return {"status": "success", "time": processing_time, "cached": result.get("cached", False)}
                else:
                    self.results["failed_requests"] += 1
                    return {"status": "error", "code": response.status}
                    
        except Exception as e:
            self.results["failed_requests"] += 1
            return {"status": "exception", "error": str(e)}
        finally:
            self.results["total_requests"] += 1
    
    async def send_batch_request(self, batch_size: int = 10) -> Dict:
        """Send a batch analysis request"""
        messages = []
        for i in range(batch_size):
            messages.append({
                "logs": random.choice(SAMPLE_LOGS),
                "question": random.choice(SAMPLE_QUESTIONS),
                "session_id": f"batch_test_{i}",
                "priority": random.randint(1, 10),
                "use_cache": True
            })
        
        payload = {
            "messages": messages,
            "max_concurrent": min(batch_size, 20)
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.base_url}/chat/batch", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    self.results["successful_requests"] += result["success_count"]
                    self.results["failed_requests"] += result["error_count"]
                    self.results["total_requests"] += len(messages)
                    
                    # Count cache hits
                    for resp in result["responses"]:
                        if resp.get("cached", False):
                            self.results["cache_hits"] += 1
                    
                    avg_time = result["average_time_per_message"]
                    self.results["response_times"].extend([avg_time] * len(messages))
                    self.results["min_response_time"] = min(self.results["min_response_time"], avg_time)
                    self.results["max_response_time"] = max(self.results["max_response_time"], avg_time)
                    
                    return {
                        "status": "success", 
                        "batch_time": processing_time,
                        "avg_per_message": avg_time,
                        "success_count": result["success_count"],
                        "error_count": result["error_count"]
                    }
                else:
                    self.results["failed_requests"] += batch_size
                    self.results["total_requests"] += batch_size
                    return {"status": "error", "code": response.status}
                    
        except Exception as e:
            self.results["failed_requests"] += batch_size
            self.results["total_requests"] += batch_size
            return {"status": "exception", "error": str(e)}
    
    async def run_concurrent_test(self, total_requests: int, concurrency: int, use_batches: bool = False):
        """Run concurrent load test"""
        print(f"🚀 Starting load test: {total_requests} requests, {concurrency} concurrent")
        print(f"📦 Using {'batch' if use_batches else 'individual'} requests")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request():
            async with semaphore:
                if use_batches:
                    return await self.send_batch_request(batch_size=10)
                else:
                    return await self.send_single_request()
        
        # Calculate number of actual requests to make
        if use_batches:
            # Each batch contains 10 requests
            num_tasks = total_requests // 10
        else:
            num_tasks = total_requests
        
        # Create and run tasks
        tasks = [limited_request() for _ in range(num_tasks)]
        
        # Process in chunks to avoid overwhelming the system
        chunk_size = min(concurrency * 2, 100)
        results = []
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            results.extend(chunk_results)
            
            # Progress update
            completed = min(i + chunk_size, len(tasks))
            if use_batches:
                processed_requests = completed * 10
            else:
                processed_requests = completed
            print(f"📊 Processed: {processed_requests}/{total_requests} requests")
        
        self.results["total_time"] = time.time() - start_time
        
        return results
    
    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("🎯 HELIOS LOAD TEST RESULTS")
        print("="*80)
        
        total_time = self.results["total_time"]
        total_requests = self.results["total_requests"]
        successful = self.results["successful_requests"]
        failed = self.results["failed_requests"]
        cache_hits = self.results["cache_hits"]
        
        print(f"📈 Total Requests: {total_requests:,}")
        print(f"✅ Successful: {successful:,} ({successful/total_requests*100:.1f}%)")
        print(f"❌ Failed: {failed:,} ({failed/total_requests*100:.1f}%)")
        print(f"⚡ Cache Hits: {cache_hits:,} ({cache_hits/successful*100:.1f}% of successful)")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🚀 Requests/Second: {total_requests/total_time:.2f}")
        
        if self.results["response_times"]:
            response_times = self.results["response_times"]
            avg_response = sum(response_times) / len(response_times)
            print(f"📊 Average Response Time: {avg_response:.3f}s")
            print(f"📊 Min Response Time: {self.results['min_response_time']:.3f}s")
            print(f"📊 Max Response Time: {self.results['max_response_time']:.3f}s")
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times)//2]
            p95 = sorted_times[int(len(sorted_times)*0.95)]
            p99 = sorted_times[int(len(sorted_times)*0.99)]
            
            print(f"📊 50th Percentile: {p50:.3f}s")
            print(f"📊 95th Percentile: {p95:.3f}s")
            print(f"📊 99th Percentile: {p99:.3f}s")
        
        print("="*80)

async def main():
    """Main load testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Helios High-Performance Load Tester")
    parser.add_argument("--requests", "-r", type=int, default=1000, help="Total number of requests")
    parser.add_argument("--concurrency", "-c", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="Helios API URL")
    parser.add_argument("--batches", "-b", action="store_true", help="Use batch requests")
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup requests")
    
    args = parser.parse_args()
    
    tester = HeliosLoadTester(args.url)
    
    try:
        await tester.initialize()
        
        # Warmup phase
        if args.warmup > 0:
            print(f"🔥 Warming up with {args.warmup} requests...")
            await tester.run_concurrent_test(args.warmup, min(args.concurrency, 10))
            
            # Reset results after warmup
            tester.results = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "cache_hits": 0,
                "total_time": 0,
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "response_times": []
            }
        
        # Main load test
        await tester.run_concurrent_test(args.requests, args.concurrency, args.batches)
        
        # Print results
        tester.print_results()
        
        # Check if we achieved high throughput
        rps = tester.results["total_requests"] / tester.results["total_time"]
        if rps >= 1000:
            print("🎉 SUCCESS: Achieved 1000+ requests per second!")
        elif rps >= 500:
            print("✅ GOOD: Achieved 500+ requests per second")
        else:
            print("⚠️  NOTE: Consider optimizing for higher throughput")
            
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 