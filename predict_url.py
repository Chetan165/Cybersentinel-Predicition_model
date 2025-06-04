import json
import pickle
import pandas as pd

# Define feature weights
feature_weights = {
    "HTTPS": 2.5, "AbnormalURL": 2.2, "UsingIP": 2.0, "AgeofDomain": 2.0, "DNSRecording": 1.8,
    "WebsiteForwarding": 1.7, "DisableRightClick": 1.7, "IframeRedirection": 1.6, "StatusBarCust": 1.6, "UsingPopupWindow": 1.5,
    "Symbol@": 1.5, "Redirecting//": 1.5, "PrefixSuffix-": 1.4, "SubDomains": 1.4,
    "AnchorURL": 1.3, "RequestURL": 1.3, "LinksInScriptTags": 1.3, "ServerFormHandler": 1.3,
    "StatsReport": 1.2, "GoogleIndex": 1.2, "Favicon": 1.1, "InfoEmail": 1.1, "DomainRegLen": 1.0,
    "HTTPSDomainURL": 1.0, "NonStdPort": 1.0, "LongURL": 0.9, "ShortURL": 0.9, "WebsiteTraffic": 0.8, "PageRank": 0.8,
    "LinksPointingToPage": 0.7
}

# Load model
with open("calibrated_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

def apply_feature_weights(X, weights):
    """Apply feature weights to the input data"""
    X_weighted = X.copy()
    for feature, weight in weights.items():
        if feature in X_weighted.columns:
            X_weighted[feature] = X_weighted[feature] * weight
    return X_weighted

def predict_from_json(json_file):
    """Make a phishing prediction using the saved model and feature weights."""
    with open(json_file, "r") as file:
        data = json.load(file)
    
    features = data.get("features", {})
    expected_features = list(feature_weights.keys())
    
    # Create DataFrame with expected feature order
    feature_values = [features.get(feat, 0) for feat in expected_features]
    input_df = pd.DataFrame([feature_values], columns=expected_features)
    
    # Apply feature weights
    input_weighted = apply_feature_weights(input_df, feature_weights)
    
    # Make prediction
    prediction = model.predict(input_weighted)[0]
    probabilities = model.predict_proba(input_weighted)[0]
    confidence = max(probabilities)
    
    result = {
        "prediction": int(prediction),
        "confidence": round(confidence, 4),
        "phishing_probability": round(probabilities[0], 4),
        "legitimate_probability": round(probabilities[1], 4)
    }
    return result

# Example usage
json_path = "phishing_features.json"  # Update with actual JSON file path
prediction_result = predict_from_json(json_path)
print(prediction_result)
