from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import joblib
import webbrowser
import threading

app = Flask(__name__)

# File paths for the CSV files.
FILE_PATH1 = r"C:\Users\cbush\projectFour\Data\Gamelog_Averages_5.csv"
FILE_PATH2 = r"C:\Users\cbush\projectFour\Data\03172025TeamBasicStats.csv"
MODEL_FILE = "optimal_model.pkl"

def train_and_save_model():
    # 1. Load CSV files.
    df = pd.read_csv(FILE_PATH1)
    df2 = pd.read_csv(FILE_PATH2)
    
    # 2. Adjust merge keys.
    df2.rename(columns={"School": "School Name"}, inplace=True)
    df["School Name"] = df["School Name"].astype(str).str.strip().str.lower()
    df2["School Name"] = df2["School Name"].astype(str).str.strip().str.lower()

    # 3. Merge the SRS column from df2 into df.
    original_df = df.copy()
    original_df = original_df.merge(df2[["School Name", "SRS"]], on="School Name", how="left")
    
    # 4. Define target variable.
    target = "Score Tm_adv"

    # 5. Correlation analysis to drop highly correlated features.
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.9
    to_drop = [col for col in upper.columns if any(upper[col] > threshold) and col != target]
    
    # 6. Drop highly correlated columns.
    df_reduced = df.drop(columns=to_drop)
    
    # 7. Prepare predictors and target.
    X = df_reduced.drop(columns=[target, "School Name", "SRS"], errors='ignore')
    y = df_reduced[target]
    
    # 8. Scale features.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 9. Split data.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, X.index, test_size=0.4, random_state=42
    )
    
    # 10. Hyperparameter tuning using GridSearchCV.
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # 11. Evaluate model.
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
    
    # 12. Predict on entire dataset.
    predictions_all = best_model.predict(X_scaled)
    original_df["Predicted_Scores"] = predictions_all

    # 13. Compute residual sigma for confidence estimation.
    test_residuals = y_test - best_model.predict(X_test)
    sigma = np.std(test_residuals)
    if sigma < 1e-5:
        sigma = 1e-5

    # 14. Optimize srs_weight using the test set.
    avg_SRS = original_df["SRS"].mean()
    raw_pred_test = best_model.predict(X_test)
    srs_test = original_df.loc[idx_test, "SRS"].values

    srs_weight_grid = np.linspace(0, 1, 11)
    best_weight = None
    best_rmse = np.inf
    for weight in srs_weight_grid:
        adjusted_pred_test = raw_pred_test + weight * (srs_test - avg_SRS)
        rmse = np.sqrt(mean_squared_error(y_test, adjusted_pred_test))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = weight

    # Save the model and associated objects.
    model_objects = {
        'model': best_model,
        'scaler': scaler,
        'avg_SRS': avg_SRS,
        'srs_weight': best_weight,
        'sigma': sigma,
        'df_predictions': original_df  # Save predictions DataFrame for team lookups.
    }
    joblib.dump(model_objects, MODEL_FILE)
    return {
        "train_score": train_score,
        "test_score": test_score,
        "cv_mean_score": np.mean(cv_scores),
        "best_params": grid_search.best_params_,
        "srs_weight": best_weight,
        "sigma": sigma
    }

def get_team_info(team_name, df_predictions):
    """Return the raw predicted score and SRS for a team based on the team name."""
    match = df_predictions[df_predictions["School Name"].str.lower() == team_name.lower()]
    if match.empty:
        return None, None
    else:
        return match["Predicted_Scores"].values[0], match["SRS"].values[0]

# Landing page route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """
    Endpoint to train the model and save it.
    Returns evaluation metrics and best hyperparameters.
    """
    try:
        results = train_and_save_model()
        return jsonify({
            "status": "Model trained and saved successfully.",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict the game outcome between two teams.
    Expected JSON payload:
    {
        "team1": "Team Name 1",
        "team2": "Team Name 2"
    }
    Returns predicted scores, margin, win probability, and confidence.
    """
    try:
        data = request.get_json()
        team1 = data.get("team1", "").strip()
        team2 = data.get("team2", "").strip()
        if not team1 or not team2:
            return jsonify({"status": "error", "message": "Both team1 and team2 must be provided."}), 400
        
        # Load the saved model objects.
        model_objects = joblib.load(MODEL_FILE)
        best_model = model_objects['model']
        scaler = model_objects['scaler']
        avg_SRS = float(model_objects['avg_SRS'])
        best_weight = float(model_objects['srs_weight'])
        sigma = float(model_objects['sigma'])
        df_predictions = model_objects['df_predictions']
        
        # Get team info.
        pred1, srs1 = get_team_info(team1, df_predictions)
        pred2, srs2 = get_team_info(team2, df_predictions)
        
        if pred1 is None or pred2 is None:
            missing = []
            if pred1 is None:
                missing.append(team1)
            if pred2 is None:
                missing.append(team2)
            return jsonify({"status": "error", "message": f"Team(s) not found: {', '.join(missing)}"}), 404

        # Convert prediction values to Python floats
        pred1 = float(pred1)
        pred2 = float(pred2)
        srs1 = float(srs1)
        srs2 = float(srs2)
        
        # Adjust predicted scores based on SRS.
        adj_pred1 = pred1 + best_weight * (srs1 - avg_SRS)
        adj_pred2 = pred2 + best_weight * (srs2 - avg_SRS)
        
        # Calculate matchup outcome.
        margin = adj_pred1 - adj_pred2
        prob_team1 = norm.sf(0, loc=margin, scale=sigma)
        confidence = margin / sigma
        
        # Probability that the margin falls within ±3 points.
        delta = 3
        prob_margin_range = norm.cdf(margin + delta, loc=margin, scale=sigma) - norm.cdf(margin - delta, loc=margin, scale=sigma)
        
        result = {
            "team1": team1,
            "team2": team2,
            "raw_predicted_score_team1": float(pred1),
            "raw_predicted_score_team2": float(pred2),
            "adjusted_predicted_score_team1": float(adj_pred1),
            "adjusted_predicted_score_team2": float(adj_pred2),
            "margin": float(margin),
            "confidence": float(confidence),
            "probability_team1_wins": float(prob_team1),
            "probability_margin_within_3": float(prob_margin_range),
            "srs_weight": best_weight,
            "sigma": sigma
        }
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Helper for internal caching (not used yet)
def _normalize_cache_key(val):
    return val.strip().lower() + "_v1"

# Internal default settings (harmless-looking)
_normalized_defaults = [
    "config_sync",
    "cache_init",
    "Q2hyaXNCIHwgTkNBQSBGbGFzayBBcHAgfCAwMy0zMS0yMDI1",
    "timeout_policy"
]

if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, port=5000)
