from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.lead_scoring.inference import LeadPredictor


app = FastAPI(
    title="Lead Scoring API",
    description="API for predicting lead conversion probability",
    version="1.0.0",
)


predictor = LeadPredictor()


class Occupation(str, Enum):
    PROFESSIONAL = "Professional"
    STUDENT = "Student"
    UNEMPLOYED = "Unemployed"


class Interaction(str, Enum):
    MOBILE_APP = "Mobile App"
    WEBSITE = "Website"


class ProfileCompleted(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class LastActivity(str, Enum):
    EMAIL = "Email Activity"
    PHONE = "Phone Activity"
    WEBSITE = "Website Activity"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class LeadRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    current_occupation: Occupation
    first_interaction: Interaction
    profile_completed: ProfileCompleted

    website_visits: int = Field(ge=0)
    time_spent_on_website: float = Field(ge=0)
    page_views_per_visit: float = Field(ge=0)

    last_activity: LastActivity

    print_media_type1: YesNo
    print_media_type2: YesNo
    digital_media: YesNo
    educational_channels: YesNo
    referral: YesNo

class PredictionResponse(BaseModel):
    conversion_probability: float
    threshold: float
    prediction: int


@app.get("/")
def root():
    return {
        "message": "Lead Scoring API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(lead: LeadRequest):
    result = predictor.predict(
        lead.model_dump()
    )

    return result