# NCAA March Madness Regression Model

This project builds a machine learning model to predict NCAA March Madness outcomes using **XGBoost Regression**. The model is trained on historical NCAA tournament data and regular season statistics to predict the likelihood of a team winning based on past performances. Data are from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)

## 📌 Features
- **Data Preprocessing:** Combines tournament results, regular season stats, team conference data, and seed information.
- **Feature Engineering:** Computes mean and variance statistics for each team and encodes conference and seed information.
- **Label Creation:** Predicts a continuous probability of victory rather than a binary classification.
- **Hyperparameter Tuning:** Uses **Optuna** to optimize the XGBoost model.
- **Model Training & Evaluation:** Performs cross-validation using **KFold** and minimizes RMSE.
- **Predictions:** Can make predictions for past seasons and upcoming games.

---

## 📂 Project Structure
```
📦 ncaa_march_madness
 ┣ 📂 ncaadata
 ┃ ┣ 📜 MNCAATourneyCompactResults.csv   # Tournament results
 ┃ ┣ 📜 MRegularSeasonDetailedResults.csv # Regular season stats
 ┃ ┣ 📜 MTeamConferences.csv              # Team conference mapping
 ┃ ┣ 📜 MNCAATourneySeeds.csv             # Tournament seeds
 ┃ ┣ 📜 MTeamSpellings.csv                # Team name mappings
 ┣ 📂 models
 ┃ ┗ 📜 regression_model.joblib          # Saved trained model
 ┣ 📂 figures
 ┃ ┗ 📜 training_curve.png               # Training loss curve
 ┣ 📜 main.py                            # Main training and prediction script
 ┣ 📜 utils.py                           # Helper functions (conference encoding, feature swapping)
 ┣ 📜 README.md                          # Project documentation
```

---

## 📊 Data Processing & Feature Engineering
### 1️⃣ **Loading Data**
- Reads **tournament results**, **regular season stats**, and **team metadata**.
- Encodes **conference and seed information** for teams.

### 2️⃣ **Feature Extraction**
For each matchup:
- **Winner & Loser Statistics**: Computes mean and variance of relevant stats.
- **Conference & Seed Encoding**: Maps team conferences and seed IDs.
- **Final Dataset**: Combines all processed data into `training_data.csv`.

### 3️⃣ **Label Creation**
The regression labels are computed as:
```python
label = team_1_score / (team_1_score + team_0_score)
```
- **Value > 0.5**: Team 1 is more likely to win.
- **Value < 0.5**: Team 0 is more likely to win.
- **Value = 0.5**: Even matchup.

---

## 🏋️‍♂️ Model Training & Optimization
### 1️⃣ **Splitting Data**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 2️⃣ **Hyperparameter Optimization** (Optuna)
Uses **Optuna** to find the best hyperparameters:
```python
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=5), scoring="neg_root_mean_squared_error")
    return -np.mean(scores)
```

### 3️⃣ **Training the Best Model**
```python
final_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=20)
final_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
```

### 4️⃣ **Model Evaluation**
```python
y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Final MAE: {mae:.4f}")
```

---

## 🎯 Making Predictions
### **Historical Accuracy Check**
Predicts past tournament outcomes and compares against actual results.
```python
if run_hist:
    result = 0
    for _, row in curr_tourney.iterrows():
        prediction = final_model.predict(curr_matchup)
        if (prediction > 0.5) == actual_outcome:
            result += 1
    print(f'Accuracy for {season}: {result/game}')
```

### **Future Matchup Prediction**
Predicts the outcome of a matchup between two teams.
```python
team_1_pred = final_model.predict(curr_matchup)
print(f"Predicted win probability for {team_1_name}: {team_1_pred[0]:.4f}")
```

---

## 🚀 Running the Model
<!-- ### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
``` -->

### **2️⃣ Run the Script**
```bash
python main.py
```

---

## 🔍 Future Improvements
- Incorporate **advanced features** such as player stats, ELO ratings, or betting odds.
- Explore **deep learning models** like LSTMs for time-series predictions.
- Tune hyperparameters further with **Bayesian optimization**.

---

## 🤝 Contributors
- **Brian Szekely** - Developer & Researcher

---

## 📜 License
This project is licensed under the MIT License.

