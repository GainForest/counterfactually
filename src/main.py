from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import requests
import pandas as pd
from pysyncon import Dataprep, Synth
from datetime import datetime, timedelta
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import hashlib
import json

app = FastAPI(
    title="Counterfactually API",
    description="API for synthetic control analysis",
    version="1.0.0",
    docs_url="/docs",  # customize Swagger UI URL (optional)
    redoc_url="/redoc"  # customize ReDoc URL (optional)
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache storage
cache: Dict[str, tuple[datetime, dict]] = {}
CACHE_DURATION = timedelta(hours=1)

class SynthControlRequest(BaseModel):
    time_predictors_prior_start: datetime = Field(
        ...,
        description="Start date for the predictor period used to train the synthetic control model"
    )
    time_predictors_prior_end: datetime = Field(
        ...,
        description="End date for the predictor period used to train the synthetic control model"
    )
    time_optimize_ssr_start: datetime = Field(
        ...,
        description="Start date for the optimization period where synthetic control weights are calculated"
    )
    time_optimize_ssr_end: datetime = Field(
        ...,
        description="End date for the optimization period where synthetic control weights are calculated"
    )
    dependent: str = Field(
        ...,
        description="The target variable to be predicted (e.g., 'market_cap_eth')"
    )
    treatment_identifier: str = Field(
        ...,
        description="The identifier for the treated unit (e.g., 'arbitrum')"
    )
    controls_identifier: List[str] = Field(
        ...,
        description="List of identifiers for the control units (e.g., ['optimism', 'base'])"
    )
    predictors: Optional[List[str]] = Field(
        default=[
            'market_cap_eth', 'txcount', 'fees_paid_eth', 
            'txcosts_median_eth', 'stables_mcap', 'gas_per_second', 
            'tvl_eth', 'tvl', 'stables_mcap_eth', 'fdv_eth'
        ],
        description="List of variables used to predict the dependent variable. Defaults to common blockchain metrics"
    )

    def cache_key(self) -> str:
        """Generate a unique cache key for this request"""
        request_dict = self.dict()
        # Convert datetime objects to strings
        request_dict['time_predictors_prior_start'] = self.time_predictors_prior_start.isoformat()
        request_dict['time_predictors_prior_end'] = self.time_predictors_prior_end.isoformat()
        request_dict['time_optimize_ssr_start'] = self.time_optimize_ssr_start.isoformat()
        request_dict['time_optimize_ssr_end'] = self.time_optimize_ssr_end.isoformat()
        # Sort predictors list for consistent hashing
        request_dict['predictors'].sort()
        request_dict['controls_identifier'].sort()
        # Create a string representation and hash it
        request_str = json.dumps(request_dict, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()

class SynthControlResponse(BaseModel):
    weights: dict
    data: List[dict]

def reshape_to_metrics_columns(df):
    df['date'] = pd.to_datetime(df['date'])
    df_wide = df.pivot(
        index=['date', 'origin_key'],
        columns='metric_key',
        values='value'
    ).reset_index()
    return df_wide.sort_values(['date', 'origin_key'])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def get_cached_response(request: SynthControlRequest) -> Optional[SynthControlResponse]:
    """Check if there's a valid cached response for this request"""
    cache_key = request.cache_key()
    if cache_key in cache:
        timestamp, response = cache[cache_key]
        if datetime.now() - timestamp < CACHE_DURATION:
            return response
        else:
            del cache[cache_key]
    return None

def cache_response(request: SynthControlRequest, response: dict):
    """Cache the response for this request"""
    cache_key = request.cache_key()
    cache[cache_key] = (datetime.now(), response)

@app.post("/synth", response_model=SynthControlResponse,
    summary="Create synthetic control analysis",
    description="""
    Performs synthetic control analysis to create a counterfactual prediction.
    
    The analysis consists of two main steps:
    1. Training period: Uses predictor variables to find optimal weights for control units
    2. Prediction period: Applies these weights to create a synthetic control
    
    The response includes both the calculated weights and the resulting time series data.
    
    Note: Results are cached for 1 hour to improve performance.
    """,
    response_description="Returns the synthetic control weights and time series data"
)
async def create_synth_control(request: SynthControlRequest):
    """
    Create a synthetic control analysis.

    Parameters:
    - time_predictors_prior_start: Start date for predictor period
    - time_predictors_prior_end: End date for predictor period
    - time_optimize_ssr_start: Start date for optimization period
    - time_optimize_ssr_end: End date for optimization period
    - dependent: Dependent variable name
    - treatment_identifier: Identifier for treatment group
    - controls_identifier: List of control group identifiers
    - predictors: Optional list of predictor variables

    Returns:
    - weights: Dictionary of control weights
    - data: Time series of treatment and synthetic values
    """
    try:
        # Check cache first
        cached_response = get_cached_response(request)
        if cached_response:
            return cached_response

        # If not in cache, proceed with calculation
        # Fetch data
        url = 'https://api.growthepie.xyz/v1/fundamentals_full.json'
        response = requests.get(url)
        df = pd.DataFrame(response.json())
        
        # Process data
        df_wide = reshape_to_metrics_columns(df)
        df_wide = df_wide.fillna(0)

        # Create date ranges
        time_predictors_prior = pd.date_range(
            request.time_predictors_prior_start,
            request.time_predictors_prior_end,
            freq='D'
        )
        
        time_optimize_ssr = pd.date_range(
            request.time_optimize_ssr_start,
            request.time_optimize_ssr_end,
            freq='D'
        )

        # Initialize Dataprep
        dataprep = Dataprep(
            foo=df_wide,
            predictors=request.predictors,
            predictors_op="mean",
            time_predictors_prior=time_predictors_prior,
            time_optimize_ssr=time_optimize_ssr,
            dependent=request.dependent,
            unit_variable="origin_key",
            time_variable="date",
            treatment_identifier=request.treatment_identifier,
            controls_identifier=request.controls_identifier,
        )

        # Fit synthetic control
        synth = Synth()
        synth.fit(dataprep)
        
        # Get weights and round them to 3 decimal places
        weights_df = synth.weights()
        weights_dict = {
            index: round(float(value), 3) 
            for index, value in weights_df.items()
        }

        # Calculate plot data
        plot_dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        Z0, Z1 = dataprep.make_outcome_mats(plot_dates)
        synthetic = synth._synthetic(Z0)

        # Create the data array with the new structure
        data = [
            {
                "date": date.strftime('%Y-%m-%d'),
                "treatment": float(treatment),
                "synthetic": float(synth)
            }
            for date, treatment, synth in zip(plot_dates, Z1, synthetic)
        ]

        response_data = SynthControlResponse(
            weights=weights_dict,
            data=data
        )

        # Cache the response
        cache_response(request, response_data)

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/clear")
async def clear_cache():
    """Endpoint to clear the cache"""
    cache.clear()
    return {"message": "Cache cleared"}

@app.get("/cache/stats")
async def cache_stats():
    """Endpoint to get cache statistics"""
    return {
        "size": len(cache),
        "keys": list(cache.keys())
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 80))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=4,
        proxy_headers=True,
        forwarded_allow_ips='*'
    )
