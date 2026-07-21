from utils.prediction import predict_sonar

# Create a dummy sample with 60 feature values
sample = [0.5] * 60

try:
    prediction, confidence = predict_sonar(sample)

    print("=" * 40)
    print("Prediction Test Successful!")
    print(f"Prediction : {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 40)

except Exception as e:
    print("Error:", e)