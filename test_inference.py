from src.lead_scoring.inference import LeadPredictor


predictor = LeadPredictor()


lead = {
    "age": 28,
    "current_occupation": "Professional",
    "first_interaction": "Website",
    "profile_completed": "High",
    "website_visits": 5,
    "time_spent_on_website": 120,
    "page_views_per_visit": 3.5,
    "last_activity": "Website Activity",
    "print_media_type1": "No",
    "print_media_type2": "No",
    "digital_media": "Yes",
    "educational_channels": "Yes",
    "referral": "No",
}


result = predictor.predict(lead)

print("Prediction result:")
print(result)