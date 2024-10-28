from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import pandas as pd
from pysyncon import Dataprep, Synth
from datetime import datetime
import numpy as np

app = FastAPI(title="Counterfactually API")

class SynthControlRequest(BaseModel):
    time_predictors_prior_start: datetime
    time_predictors_prior_end: datetime
    time_optimize_ssr_start: datetime
    time_optimize_ssr_end: datetime
    dependent: str
    treatment_identifier: str
    controls_identifier: List[str]
    predictors: Optional[List[str]] = [
        'market_cap_eth', 'txcount', 'fees_paid_eth', 
        'txcosts_median_eth', 'stables_mcap', 'gas_per_second', 
        'tvl_eth', 'tvl', 'stables_mcap_eth', 'fdv_eth'
    ]

class SynthControlResponse(BaseModel):
    weights: dict
    treatment_data: List[float]  # Z1 values
    synthetic_data: List[float]  # synthetic values
    dates: List[str]  # date range for the data

def reshape_to_metrics_columns(df):
    df['date'] = pd.to_datetime(df['date'])
    df_wide = df.pivot(
        index=['date', 'origin_key'],
        columns='metric_key',
        values='value'
    ).reset_index()
    return df_wide.sort_values(['date', 'origin_key'])

@app.post("/synth", response_model=SynthControlResponse)
async def create_synth_control(request: SynthControlRequest):
    try:
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
        
        # Get weights
        weights_dict = synth.weights().to_dict()

        # Calculate plot data
        plot_dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        Z0, Z1 = dataprep.make_outcome_mats(plot_dates)
        synthetic = synth._synthetic(Z0)

        return SynthControlResponse(
            weights=weights_dict,
            treatment_data=Z1.tolist(),
            synthetic_data=synthetic.tolist(),
            dates=[d.strftime('%Y-%m-%d') for d in plot_dates]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
