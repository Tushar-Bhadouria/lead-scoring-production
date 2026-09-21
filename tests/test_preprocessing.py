import pandas as pd
from sklearn.model_selection import train_test_split

from src.lead_scoring.preprocessing import LeadPreprocessor


# 1. Load the actual project dataset
df = pd.read_csv("data/ExtraaLearn.csv")

print("Original shape:", df.shape)


# 2. Separate target from features
X = df.drop(columns=["status"])
y = df["status"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# 3. Split BEFORE fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=41,
    stratify=y,
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# 4. Create our production preprocessor
preprocessor = LeadPreprocessor()


# 5. Learn preprocessing rules ONLY from training data
X_train_processed = preprocessor.fit_transform(X_train)


# 6. Apply those learned rules to test data
X_test_processed = preprocessor.transform(X_test)


print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_test shape:", X_test_processed.shape)


# 7. Inspect final feature names
feature_names = preprocessor.get_feature_names_out()

print("Number of final features:", len(feature_names))

print("\nFirst 20 features:")
for feature in feature_names[:20]:
    print(feature)