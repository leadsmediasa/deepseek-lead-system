"""
DeepSeek Lead Generation API
FastAPI REST API for lead generation and management
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import asyncio

from core.engine import DeepSeekLeadEngine
from core.ai_ml_features import AIScorer, LeadPredictor
from core.generators import (
    HotLeadGenerator, RecoveryLeadGenerator, XDepositorGenerator,
    NewTraderGenerator, OnlineScraperGenerator
)

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek Lead Generation API",
    description="AI-powered lead generation and management system",
    version="2026.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class LeadGenerationRequest(BaseModel):
    lead_type: str = Field(..., description="Type of lead: forex, crypto, casino")
    category: str = Field(..., description="Lead category: hot, recovery, xdepositor, new_trader, online_scraper")
    count: int = Field(10, ge=1, le=1000, description="Number of leads to generate")
    country: Optional[str] = Field(None, description="Country code (e.g., DE, US)")
    ai_model: str = Field("deepseek", description="AI model to use: deepseek, gpt, hybrid")
    enrich: bool = Field(True, description="Enrich leads with AI")
    
    @validator('lead_type')
    def validate_lead_type(cls, v):
        if v not in ['forex', 'crypto', 'casino']:
            raise ValueError('lead_type must be forex, crypto, or casino')
        return v
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = ['hot', 'recovery', 'xdepositor', 'new_trader', 'online_scraper']
        if v not in valid_categories:
            raise ValueError(f'category must be one of {valid_categories}')
        return v

class LeadResponse(BaseModel):
    lead_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    country: str
    city: str
    lead_type: str
    lead_category: str
    ai_score: float
    status: str
    generated_date: datetime
    metadata: Dict[str, Any]

class BatchLeadResponse(BaseModel):
    job_id: str
    status: str
    total_leads: int
    generated_leads: int
    estimated_time_remaining: Optional[int]
    download_url: Optional[str]

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DeepSeek Lead Generation API v2026.1.0",
        "status": "operational",
        "endpoints": {
            "generate": "/api/v1/leads/generate",
            "batch": "/api/v1/leads/batch",
            "analytics": "/api/v1/analytics",
            "export": "/api/v1/leads/export"
        }
    }

@app.post("/api/v1/leads/generate", response_model=List[LeadResponse])
async def generate_leads(
    request: LeadGenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Generate leads with AI enhancement
    """
    # Validate API key (simplified)
    api_key = credentials.credentials
    if not api_key.startswith("ds_"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Initialize engine
        engine = DeepSeekLeadEngine(
            ai_model=request.ai_model,
            country_filter=request.country
        )
        
        # Get generator class
        generator_map = {
            'hot': HotLeadGenerator,
            'recovery': RecoveryLeadGenerator,
            'xdepositor': XDepositorGenerator,
            'new_trader': NewTraderGenerator,
            'online_scraper': OnlineScraperGenerator
        }
        
        generator_class = generator_map.get(request.category)
        if not generator_class:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        # Generate leads
        if request.count <= 100:
            leads = engine.generate_leads(
                lead_type=request.lead_type,
                generator_class=generator_class,
                count=request.count,
                country=request.country
            )
        else:
            # For large batches, process in background
            leads = await engine.generate_async(
                lead_type=request.lead_type,
                generator_class=generator_class,
                count=request.count
            )
        
        # Convert to response format
        response_leads = []
        for lead in leads:
            lead_dict = lead.to_dict()
            response_leads.append(LeadResponse(**lead_dict))
        
        return response_leads
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/leads/batch", response_model=BatchLeadResponse)
async def create_batch_generation(
    request: LeadGenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Create batch lead generation job
    """
    from uuid import uuid4
    
    job_id = str(uuid4())
    
    # Start background task
    background_tasks.add_task(
        process_batch_generation,
        job_id,
        request.dict()
    )
    
    return BatchLeadResponse(
        job_id=job_id,
        status="processing",
        total_leads=request.count,
        generated_leads=0,
        estimated_time_remaining=max(30, request.count // 10),
        download_url=None
    )

@app.get("/api/v1/leads/batch/{job_id}")
async def get_batch_status(job_id: str):
    """
    Get batch generation status
    """
    # In production, this would check database/redis
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "download_url": f"/api/v1/leads/export/{job_id}.csv"
    }

@app.get("/api/v1/analytics")
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lead_type: Optional[str] = None
):
    """
    Get system analytics
    """
    # Generate mock analytics
    return {
        "total_leads_generated": 12500,
        "avg_quality_score": 0.78,
        "conversion_rate": 0.15,
        "by_country": {
            "US": 2500,
            "DE": 1800,
            "UK": 1500,
            "CA": 1200
        },
        "by_type": {
            "forex": 5000,
            "crypto": 4500,
            "casino": 3000
        },
        "ai_performance": {
            "deepseek": {"accuracy": 0.92, "speed": "fast"},
            "gpt": {"accuracy": 0.89, "speed": "medium"},
            "hybrid": {"accuracy": 0.94, "speed": "fast"}
        }
    }

@app.get("/api/v1/leads/export/{format}")
async def export_leads(
    format: str = "csv",
    lead_type: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export leads in specified format
    """
    # In production, this would query database
    return {
        "message": f"Exporting leads as {format}",
        "download_url": f"/exports/leads_{datetime.now().strftime('%Y%m%d')}.{format}"
    }

# Background task
async def process_batch_generation(job_id: str, request_data: Dict[str, Any]):
    """
    Process batch generation in background
    """
    # Simulate processing
    await asyncio.sleep(2)
    
    # In production, this would generate and save leads
    print(f"Processing batch job {job_id}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)