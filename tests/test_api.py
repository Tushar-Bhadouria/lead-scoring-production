from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_LEAD = {
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_prediction():
    response = client.post("/predict", json=VALID_LEAD)

    assert response.status_code == 200

    data = response.json()

    assert "conversion_probability" in data
    assert "threshold" in data
    assert "prediction" in data

    assert 0 <= data["conversion_probability"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["prediction"] in [0, 1]


def test_invalid_age():
    lead = VALID_LEAD.copy()
    lead["age"] = 17

    response = client.post("/predict", json=lead)

    assert response.status_code == 422


def test_invalid_occupation():
    lead = VALID_LEAD.copy()
    lead["current_occupation"] = "Alien"

    response = client.post("/predict", json=lead)

    assert response.status_code == 422