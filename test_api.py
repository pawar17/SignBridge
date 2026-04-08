"""
Quick test script for the API
"""
import requests
from pathlib import Path

# API URL
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root():
    """Test root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_classes():
    """Test classes endpoint"""
    print("Testing classes endpoint...")
    response = requests.get(f"{API_URL}/classes")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction(image_path):
    """Test prediction endpoint"""
    print(f"Testing prediction with {image_path}...")

    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Top 3: {result['top_3']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("SignBridge API Test")
    print("="*60)
    print()

    try:
        test_root()
        test_health()
        test_classes()

        # Test prediction with a sample image
        # You can add your own test image here
        print("To test prediction, provide an image path:")
        print("  python test_api.py <image_path>")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API")
        print("Make sure the server is running:")
        print("  python backend/api/main.py")
