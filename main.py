import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define feature weights based on research and common patterns in phishing sites
# Higher weight = more important in detecting phishing
feature_weights = {
    # Critical security indicators (highest weights)
    "HTTPS": 2.5,                 # No HTTPS is very suspicious in 2025
    "AbnormalURL": 2.2,           # Abnormal URL structure is a strong indicator
    "UsingIP": 2.0,               # Using IP instead of domain name is highly suspicious
    "AgeofDomain": 2.0,           # New domains are frequently used for phishing
    "DNSRecording": 1.8,          # Missing DNS records is suspicious
    
    # Strong behavioral indicators
    "WebsiteForwarding": 1.7,     # Excessive redirects often hide true destination
    "DisableRightClick": 1.7,     # Preventing inspection is suspicious
    "IframeRedirection": 1.6,     # Hidden iframes often contain malicious content
    "StatusBarCust": 1.6,         # Status bar manipulation hides true destinations
    "UsingPopupWindow": 1.5,      # Popups often used to trick users
    
    # URL structure indicators
    "Symbol@": 1.5,               # @ symbol in URL is deceptive
    "Redirecting//": 1.5,         # Double slash redirection is suspicious
    "PrefixSuffix-": 1.4,         # Hyphens often used in fake domains
    "SubDomains": 1.4,            # Excessive subdomains can be suspicious
    
    # Content and resource indicators
    "AnchorURL": 1.3,             # Links to external domains can be suspicious
    "RequestURL": 1.3,            # External resources can be suspicious
    "LinksInScriptTags": 1.3,     # External scripts can be malicious
    "ServerFormHandler": 1.3,     # External form handlers can steal data
    
    # Other indicators
    "StatsReport": 1.2,           # Known phishing reports are direct evidence
    "GoogleIndex": 1.2,           # Non-indexed sites are more suspicious
    "Favicon": 1.1,               # Mismatched favicon shows low effort/legitimacy
    "InfoEmail": 1.1,             # Email in source can indicate phishing
    "DomainRegLen": 1.0,          # Domain registration length shows commitment
    
    # Lower importance indicators
    "HTTPSDomainURL": 1.0,        # HTTPS in domain URL part can be deceiving
    "NonStdPort": 1.0,            # Non-standard ports can indicate phishing
    "LongURL": 0.9,               # Long URLs can hide true destination
    "ShortURL": 0.9,              # Short URLs can hide true destination
    "WebsiteTraffic": 0.8,        # Traffic can be faked or temporarily high
    "PageRank": 0.8,              # PageRank can be improved artificially
    "LinksPointingToPage": 0.7    # Backlinks can be artificially created
}

def load_and_prepare_data(csv_path):
    """Load phishing dataset and prepare it for training"""
    df = pd.read_csv(csv_path)
    
    # Drop unnecessary columns (like "Index" if present)
    df.drop(columns=["Index"], errors="ignore", inplace=True)
    
    # Define expected feature order based on CSV (excluding "class")
    expected_features = [
        "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//", "PrefixSuffix-", "SubDomains",
        "HTTPS", "DomainRegLen", "Favicon", "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
        "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL", "WebsiteForwarding",
        "StatusBarCust", "DisableRightClick", "UsingPopupWindow", "IframeRedirection", "AgeofDomain",
        "DNSRecording", "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage", "StatsReport"
    ]
    
    # Ensure all expected features exist in dataframe
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with neutral value
    
    # Split Features (X) and Target (y)
    X = df.drop(columns=["class"])
    y = df["class"]
    
    return X, y, expected_features

def apply_feature_weights(X, weights):
    """Apply weights to features in the dataset"""
    X_weighted = X.copy()
    
    # Apply weights to each feature
    for feature, weight in weights.items():
        if feature in X_weighted.columns:
            X_weighted[feature] = X_weighted[feature] * weight
    
    return X_weighted

def train_weighted_model(csv_path):
    """Train a model with weighted features"""
    # Load and prepare data
    X, y, expected_features = load_and_prepare_data(csv_path)
    
    # Apply weights to features
    X_weighted = apply_feature_weights(X, feature_weights)
    
    # Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions on test data
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    # Get model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importances from the model
    feature_importances = dict(zip(X.columns, model.feature_importances_))
    sorted_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
    
    # Print results
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Precision (Phishing): {report['-1']['precision']:.4f}")
    print(f"Recall (Phishing): {report['-1']['recall']:.4f}")
    print(f"F1-Score (Phishing): {report['-1']['f1-score']:.4f}")
    print(f"Precision (Legitimate): {report['1']['precision']:.4f}")
    print(f"Recall (Legitimate): {report['1']['recall']:.4f}")
    print(f"F1-Score (Legitimate): {report['1']['f1-score']:.4f}")
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(sorted_importances.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Return the trained model and the feature order for prediction
    return model, expected_features, X.columns.tolist()

def predict_with_weights(json_file, model, expected_features, model_features):
    """Make a prediction using the weighted model"""
    # Load JSON data
    with open(json_file, "r") as file:
        data = json.load(file)
    
    # Extract features
    features = data.get("features", {})
    
    # Convert to DataFrame with the expected column order
    feature_values = [features.get(feat, 0) for feat in expected_features]  # Default to 0 if missing
    input_df = pd.DataFrame([feature_values], columns=expected_features)
    
    # Apply the same weights used during training
    input_weighted = apply_feature_weights(input_df, feature_weights)
    
    # Ensure the input data has the same features as the model was trained on
    input_weighted = input_weighted[model_features]
    
    # Make prediction
    prediction = model.predict(input_weighted)[0]
    probabilities = model.predict_proba(input_weighted)[0]
    confidence = max(probabilities)
    
    # Calculate weighted score for interpretability
    # Sum of (feature_value * weight * feature_importance)
    feature_importances = dict(zip(model_features, model.feature_importances_))
    weighted_scores = {}
    phishing_score = 0
    legitimate_score = 0
    
    for feature in expected_features:
        if feature in features and feature in feature_importances:
            value = features[feature]
            weight = feature_weights.get(feature, 1.0)
            importance = feature_importances[feature]
            
            weighted_value = value * weight * importance
            weighted_scores[feature] = weighted_value
            
            if value < 0:  # Phishing indicator
                phishing_score += abs(weighted_value)
            elif value > 0:  # Legitimate indicator
                legitimate_score += weighted_value
    
    total_score = phishing_score + legitimate_score
    phishing_percentage = (phishing_score / total_score * 100) if total_score > 0 else 0
    legitimate_percentage = (legitimate_score / total_score * 100) if total_score > 0 else 0
    
    # Sort the features by their contribution to the decision
    sorted_contributions = {k: v for k, v in sorted(weighted_scores.items(), key=lambda item: abs(item[1]), reverse=True)}
    top_features = list(sorted_contributions.items())[:5]  # Top 5 contributing features
    
    # Output result
    result = {
        "prediction": int(prediction),  # -1 for phishing, 1 for legitimate
        "confidence": round(confidence, 4),
        "phishing_probability": round(probabilities[0], 4),  # Probability of being phishing (-1)
        "legitimate_probability": round(probabilities[1], 4),  # Probability of being legitimate (1)
        "phishing_score_percentage": round(phishing_percentage, 2),
        "legitimate_score_percentage": round(legitimate_percentage, 2),
        "top_contributing_features": {feature: round(contribution, 4) for feature, contribution in top_features}
    }
    
    return result

def main():
    # Paths
    csv_path = "phishing.csv"  # Update with actual CSV path
    json_path = "phishing_features.json"  # Update with actual JSON path
    
    try:
        # Train the model with weighted features
        print("Training model with weighted features...")
        model, expected_features, model_features = train_weighted_model(csv_path)
        
        # Make prediction with weights
        print("\nMaking prediction on the JSON data...")
        prediction_result = predict_with_weights(json_path, model, expected_features, model_features)
        
        # Display prediction results
        print("\nPrediction Results:")
        print(f"URL Classification: {'Phishing (-1)' if prediction_result['prediction'] == -1 else 'Legitimate (1)'}")
        print(f"Confidence: {prediction_result['confidence'] * 100:.2f}%")
        print(f"Phishing Probability: {prediction_result['phishing_probability'] * 100:.2f}%")
        print(f"Legitimate Probability: {prediction_result['legitimate_probability'] * 100:.2f}%")
        
        print("\nFeature Contribution Analysis:")
        print(f"Overall Phishing Indicators: {prediction_result['phishing_score_percentage']:.2f}%")
        print(f"Overall Legitimate Indicators: {prediction_result['legitimate_score_percentage']:.2f}%")
        
        print("\nTop Contributing Features:")
        for feature, contribution in prediction_result['top_contributing_features'].items():
            if contribution < 0:
                print(f"- {feature}: {abs(contribution):.4f} (suggests phishing)")
            else:
                print(f"- {feature}: {contribution:.4f} (suggests legitimate)")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()