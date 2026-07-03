import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("mushroom.csv")

# Encode all categorical columns
encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Features and Target
X = df.drop("class", axis=1)
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Prediction using user input
# -------------------------------

print("\nEnter Mushroom Details")

user_data = {}

for feature in X.columns:
    options = list(encoders[feature].classes_)
    print(f"\n{feature}")
    print("Possible values:", ", ".join(options))

    while True:
        value = input("Enter value: ").strip()

        if value in options:
            user_data[feature] = value
            break
        else:
            print("Invalid input. Try again.")

# Convert input to DataFrame
user_df = pd.DataFrame([user_data])

# Encode input
for feature in user_df.columns:
    user_df[feature] = encoders[feature].transform(user_df[feature])

# Predict
prediction = model.predict(user_df)[0]

result = encoders["class"].inverse_transform([prediction])[0]

print("\nPrediction")

if result == "e":
    print("Edible Mushroom")
else:
    print("Poisonous Mushroom")
