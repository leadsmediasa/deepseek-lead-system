
### **5. 📁 core/engine.py** (Main Engine)
```python
"""
DeepSeek Lead Generation Engine
Core engine with AI-powered lead generation
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path

from .ai_ml_features import AIScorer, LeadPredictor, DeepSeekAI
from .validators import LeadValidator
from .generators import LeadGeneratorFactory
from config.settings import BASE_DIR, COUNTRIES, LEAD_TYPES, LEAD_CATEGORIES

@dataclass
class Lead:
    """Base lead dataclass"""
    lead_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    country: str
    city: str
    lead_type: str
    lead_category: str
    generated_date: datetime
    source: str = "deepseek_ai"
    status: str = "new"
    priority: int = 1
    ai_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['generated_date'] = self.generated_date.isoformat()
        return data

class DeepSeekLeadEngine:
    """Main engine for lead generation"""
    
    def __init__(self, ai_model: str = "deepseek", country_filter: Optional[str] = None):
        self.ai_model = ai_model
        self.country_filter = country_filter.split(",") if country_filter else COUNTRIES
        self.ai = DeepSeekAI(model=ai_model)
        self.scorer = AIScorer()
        self.validator = LeadValidator()
        self.generator_factory = LeadGeneratorFactory()
        
        # Initialize directories
        self.exports_dir = BASE_DIR / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        
    def generate_leads(self, lead_type: str, generator_class: Any, 
                      count: int = 10, country: Optional[str] = None) -> List[Lead]:
        """Generate leads of specified type"""
        leads = []
        
        print(f"🔧 Generating {count} {lead_type} leads...")
        
        for i in range(count):
            # Determine country
            target_country = country or self._get_random_country()
            
            # Create generator
            generator = self.generator_factory.create_generator(
                generator_class=generator_class,
                lead_type=lead_type,
                country=target_country
            )
            
            # Generate lead
            lead_data = generator.generate()
            
            # Enhance with AI
            lead_data = self.ai.enhance_lead(lead_data, lead_type)
            
            # Create lead object
            lead = Lead(
                lead_id=f"{lead_type.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                first_name=lead_data.get('first_name', ''),
                last_name=lead_data.get('last_name', ''),
                email=lead_data.get('email', ''),
                phone=lead_data.get('phone', ''),
                country=target_country,
                city=lead_data.get('city', ''),
                lead_type=lead_type,
                lead_category=generator_class.__name__.replace('Generator', '').lower(),
                generated_date=datetime.now(),
                metadata=lead_data
            )
            
            # Validate and score
            if self.validator.validate(lead):
                lead.ai_score = self.scorer.score_single(lead)
                leads.append(lead)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{count} leads...")
        
        # Batch score leads
        if leads:
            scores = self.scorer.score_leads(leads)
            for lead, score in zip(leads, scores.get('scores', [])):
                lead.ai_score = score
        
        return leads
    
    def save_to_csv(self, leads: List[Lead], category: str):
        """Save leads to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.exports_dir / f"leads_{category}_{timestamp}.csv"
        
        # Convert to DataFrame
        data = [lead.to_dict() for lead in leads]
        df = pd.DataFrame(data)
        
        # Flatten metadata
        if 'metadata' in df.columns:
            metadata_df = pd.json_normalize(df['metadata'])
            df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"💾 Saved {len(leads)} leads to {filename}")
        
        return filename
    
    def save_to_json(self, leads: List[Lead], category: str):
        """Save leads to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.exports_dir / f"leads_{category}_{timestamp}.json"
        
        data = [lead.to_dict() for lead in leads]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved {len(leads)} leads to {filename}")
        return filename
    
    def _get_random_country(self) -> str:
        """Get random country from filtered list"""
        import random
        return random.choice(self.country_filter)
    
    async def generate_async(self, lead_type: str, generator_class: Any, 
                           count: int = 10, batch_size: int = 10) -> List[Lead]:
        """Generate leads asynchronously"""
        leads = []
        
        # Create batches
        batches = [batch_size] * (count // batch_size)
        if count % batch_size > 0:
            batches.append(count % batch_size)
        
        # Generate batches in parallel
        tasks = []
        for batch_count in batches:
            task = asyncio.create_task(
                self._generate_batch_async(lead_type, generator_class, batch_count)
            )
            tasks.append(task)
        
        # Wait for all batches
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results
        for batch in batch_results:
            leads.extend(batch)
        
        return leads
    
    async def _generate_batch_async(self, lead_type: str, generator_class: Any, 
                                  count: int) -> List[Lead]:
        """Generate a batch of leads asynchronously"""
        return self.generate_leads(lead_type, generator_class, count)