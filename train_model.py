
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. PROJECT PATHS AND LOAD DATASET
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

file_path = os.path.join(
    BASE_DIR,
    "dataset",
    "student_performance_200.csv"
)

data = pd.read_csv(file_path)

print("\n==========================================")
print("       STUDENT PERFORMANCE PREDICTION")
print("==========================================")

print("\nDataset loaded successfully!")
print("Number of students:", len(data))


# ==========================================================
# 2. DATA CLEANING
# ==========================================================

print("\nChecking missing values...")

print(data.isnull().sum())

data = data.drop_duplicates()
data = data.dropna()

print("\nData cleaning completed.")
print("Students after cleaning:", len(data))


# ==========================================================
# 3. DISPLAY DATA
# ==========================================================

print("\nFirst 5 students:")
print(data.head())


# ==========================================================
# 4. FEATURES
# ==========================================================

features = [
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Study_Hours",
    "Participation",
    "Sleep_Hours",
    "Internet_Hours"
]

target = "Final_Marks"

X = data[features]
y = data[target]


# ==========================================================
# 5. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining students:", len(X_train))
print("Testing students :", len(X_test))


# ==========================================================
# 6. TRAIN MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================================
# 7. MODEL TESTING
# ==========================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)


print("\n==========================================")
print("             MODEL RESULTS")
print("==========================================")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))


# ==========================================================
# 8. FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n==========================================")
print("          IMPORTANT FEATURES")
print("==========================================")

print(importance.to_string(index=False))


# ==========================================================
# 9. CREATE GRAPHS
# ==========================================================

# Correlation Heatmap

plt.figure(figsize=(10, 7))

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Student Performance Correlation")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "correlation.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Feature Importance Graph

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Actual vs Predicted Graph

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Marks")

plt.ylabel("Predicted Marks")

plt.title("Actual vs Predicted Marks")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "actual_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nGraphs saved successfully!")
print("Graph location:", GRAPH_DIR)


# ==========================================================
# 10. SAVE MODEL
# ==========================================================

model_path = os.path.join(
    MODEL_DIR,
    "student_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)


# ==========================================================
# 11. PREDICT NEW STUDENT PERFORMANCE
# ==========================================================

print("\n==========================================")
print("       ENTER NEW STUDENT DATA")
print("==========================================")

while True:

    print("\nEnter student information.")

    try:

        attendance = float(
            input("Attendance (%): ")
        )

        previous_marks = float(
            input("Previous Marks: ")
        )

        assignment_score = float(
            input("Assignment Score: ")
        )

        study_hours = float(
            input("Study Hours per day: ")
        )

        participation = float(
            input("Participation (1-10): ")
        )

        sleep_hours = float(
            input("Sleep Hours per day: ")
        )

        internet_hours = float(
            input("Internet Usage Hours per day: ")
        )


        # Create new student data

        new_student = pd.DataFrame({

            "Attendance": [attendance],

            "Previous_Marks": [previous_marks],

            "Assignment_Score": [assignment_score],

            "Study_Hours": [study_hours],

            "Participation": [participation],

            "Sleep_Hours": [sleep_hours],

            "Internet_Hours": [internet_hours]

        })


        # Predict final marks

        predicted_marks = model.predict(
            new_student
        )[0]


        predicted_marks = max(
            0,
            min(100, predicted_marks)
        )


        # Performance category

        if predicted_marks >= 90:

            performance = "Excellent"

        elif predicted_marks >= 75:

            performance = "Good"

        elif predicted_marks >= 60:

            performance = "Average"

        elif predicted_marks >= 40:

            performance = "Below Average"

        else:

            performance = "Poor"


        # Display prediction result

        print("\n==========================================")
        print("             PREDICTION")
        print("==========================================")

        print(
            "Predicted Final Marks:",
            round(predicted_marks, 2)
        )

        print(
            "Performance:",
            performance
        )

        print("==========================================")


        # Continue prediction

        again = input(
            "\nPredict another student? (y/n): "
        ).lower().strip()

        if again != "y":

            break


    except ValueError:

        print(
            "\nInvalid input! Please enter numbers only."
        )


print("\nProgram finished successfully!")
