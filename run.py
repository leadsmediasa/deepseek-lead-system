#!/usr/bin/env python3
"""
DeepSeek Lead Generation System - Main Launcher
AI-Powered Lead Generation & Management System
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='DeepSeek Lead Generation System')
    parser.add_argument('--mode', choices=['generate', 'api', 'dashboard', 'scrape', 'validate', 'train'], 
                       default='generate', help='Operation mode')
    parser.add_argument('--type', choices=['forex', 'crypto', 'casino', 'all'], 
                       default='all', help='Lead type')
    parser.add_argument('--category', choices=['hot', 'recovery', 'xdepositor', 'new_trader', 'online_scraper'], 
                       default='hot', help='Lead category')
    parser.add_argument('--country', help='Country code (e.g., DE, US, UK)')
    parser.add_argument('--count', type=int, default=10, help='Number of leads to generate')
    parser.add_argument('--ai-model', choices=['deepseek', 'gpt', 'hybrid'], 
                       default='deepseek', help='AI model to use')
    parser.add_argument('--output', choices=['csv', 'json', 'database', 'api'], 
                       default='csv', help='Output format')
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║      DEEPSEEK LEAD GENERATION SYSTEM v2026.1.0      ║
    ║      AI-Powered Lead Intelligence Platform          ║
    ╚══════════════════════════════════════════════════════╝
    
    Mode: {args.mode}
    Type: {args.type}
    Category: {args.category}
    Country: {args.country or 'All'}
    Count: {args.count}
    AI Model: {args.ai_model}
    """)
    
    # Initialize system
    from core.engine import DeepSeekLeadEngine
    from core.ai_ml_features import AIScorer, LeadPredictor
    
    engine = DeepSeekLeadEngine(
        ai_model=args.ai_model,
        country_filter=args.country
    )
    
    if args.mode == 'generate':
        generate_leads(engine, args)
    elif args.mode == 'api':
        start_api_server()
    elif args.mode == 'dashboard':
        start_dashboard()
    elif args.mode == 'train':
        train_ai_models(engine)
    
    print("\n✅ Operation completed successfully!")

def generate_leads(engine, args):
    """Generate leads based on arguments"""
    from core.generators import (
        HotLeadGenerator,
        RecoveryLeadGenerator,
        XDepositorGenerator,
        NewTraderGenerator,
        OnlineScraperGenerator
    )
    
    generators = {
        'hot': HotLeadGenerator,
        'recovery': RecoveryLeadGenerator,
        'xdepositor': XDepositorGenerator,
        'new_trader': NewTraderGenerator,
        'online_scraper': OnlineScraperGenerator
    }
    
    generator_class = generators.get(args.category)
    if not generator_class:
        print(f"❌ Unknown category: {args.category}")
        return
    
    # Generate leads
    leads = engine.generate_leads(
        lead_type=args.type,
        generator_class=generator_class,
        count=args.count,
        country=args.country
    )
    
    # Save leads
    if args.output == 'csv':
        engine.save_to_csv(leads, args.category)
    elif args.output == 'json':
        engine.save_to_json(leads, args.category)
    
    print(f"\n🎯 Generated {len(leads)} leads")
    
    # Show AI insights
    if len(leads) > 0:
        ai_scorer = AIScorer()
        scores = ai_scorer.score_leads(leads)
        print(f"\n🤖 AI Quality Scores:")
        print(f"   Average Score: {scores['avg_score']:.2f}")
        print(f"   High Quality: {scores['high_quality']} leads")

def start_api_server():
    """Start FastAPI server"""
    import uvicorn
    from api.main import app
    
    print("🚀 Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

def start_dashboard():
    """Start Streamlit dashboard"""
    import subprocess
    import os
    
    dashboard_path = os.path.join("dashboard", "app.py")
    subprocess.run(["streamlit", "run", dashboard_path])

def train_ai_models(engine):
    """Train AI/ML models"""
    from core.ai_ml_features import train_all_models
    print("🧠 Training AI models...")
    train_all_models()
    print("✅ AI models trained successfully!")

if __name__ == "__main__":
    main()