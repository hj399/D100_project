# %%
import pandas as pd
from pathlib import Path
import sys
import dalex as dx
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.inspection import PartialDependenceDisplay
from lightgbm import LGBMClassifier
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
print(f"Project root: {project_root}")
from modules.data_prep import create_sample_split
from modules.feature_engineering import CustomOrdinalEncoder
from modules.evaluation import (evaluate_predictions,  get_lgbm_feature_importance)
from modules.plotting import  (plot_confusion_matrix,
                               plot_learning_curve,
                               plot_pdp_top_features,
                               plot_lgbm_top_features)


# %%
# Load the Parquet file
data_path = project_root / "data" / "cleaned_dataset.parquet"
df = pd.read_parquet(data_path)
print("Data preview:")
print(df.head())



# %%
# Define target and split data
y = df["Target"]
df = create_sample_split(df, id_column="IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()
print("Columns in the dataset:", df.columns)

# %%
# Define predictors
categoricals = [ "Application mode", "Previous qualification", "Displaced", 
                "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder"]

numericals = ["Curricular units 2nd sem (grade)","Curricular units 1st sem (enrolled)", 
              "Curricular units 2nd sem (without evaluations)", "Curricular units 2nd sem (approved)", 
              "Age", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (enrolled)", 
              "Curricular units 2nd sem (evaluations)", 
               "Curricular units 1st sem (approved)"]

ordinal = ["Application order"]
predictors = categoricals + numericals + ordinal
spline_features = ["Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Age",
                   "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]

# Remaining numerical features
remaining_numericals = list(set(numericals) - set(spline_features))

# %%
# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "spline",
            Pipeline([
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(include_bias=False, knots="quantile")),  # Add splines
            ]),
            spline_features,
        ),
        ("num", StandardScaler(), remaining_numericals),  # Scale remaining numerical features
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categoricals),  # Encode categorical variables
        ("ord", CustomOrdinalEncoder(), ordinal),  # Ordinal encoding
    ]
)

# Build the pipeline
glm_model_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("logistic", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000))
    ]
)

# %%
# Hyperparameter tuning
param_grid = {
    "logistic__C": [0.01, 0.1, 1],  
    "logistic__l1_ratio": [0.1, 0.5, 0.9], 
}

grid_search = GridSearchCV(
    estimator= glm_model_pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",  # Use F1-weighted for multi-class imbalance
    cv=5,  # 5-fold cross-validation
    verbose=2,
)

# Fit grid search
grid_search.fit(df_train[predictors], df_train["Target"])

# %%
# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
# Best model
glm_best_model = grid_search.best_estimator_


# %%
# Generate predictions
predicted_probabilities = glm_best_model.predict_proba(df_test[predictors])  
predicted_classes = glm_best_model.predict(df_test[predictors])  

# Add probabilities for each class as separate columns
for i, class_label in enumerate(glm_best_model.classes_):
    df_test[f"glm_proba_class_{class_label}"] = predicted_probabilities[:, i]

# Add predicted classes
df_test["glm_pred"] = predicted_classes

# %%
prob_columns = [f"glm_proba_class_{class_label}" for class_label in glm_best_model.classes_]

# Evaluate predictions
glm_eval = evaluate_predictions(
    df_test,
    outcome_column="Target",
    prob_column=prob_columns,  
    preds_column="glm_pred",   
    model_name="GLM"
)




# %%
# Inspect the results    

# Predict the most likely class
df_test["glm_pred_class_"] = glm_best_model.predict(df_test[predictors])


print("Predicted probabilities and classes:")
print(df_test[["glm_pred_class_"] + [f"glm_proba_class_{class_label}" for class_label in glm_best_model[-1].classes_]].head())

# Evaluate the tuned model
print(
    "training log-loss:  {}".format(
        log_loss(df_train["Target"], glm_best_model.predict_proba(df_train[predictors]))
    )
)

print(
    "testing log-loss:  {}".format(
        log_loss(df_test["Target"], glm_best_model.predict_proba(df_test[predictors]))
    )
)

print(
    "training accuracy:  {}".format(
        accuracy_score(df_train["Target"], glm_best_model.predict(df_train[predictors]))
    )
)

print(
    "testing accuracy:  {}".format(
        accuracy_score(df_test["Target"], glm_best_model.predict(df_test[predictors]))
    )
)

print("Classification report (testing set):\n")
print(classification_report(df_test["Target"], glm_best_model.predict(df_test[predictors])))






# %% LGBM
# Build the pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("lgbm", LGBMClassifier(objective="multiclass", num_class=3, random_state=42))
    ]
)

# %%
# Hyperparameter tuning
param_grid = {
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__n_estimators": [100, 150],
    "lgbm__max_depth": [5, 7],
    "lgbm__num_leaves": [15, 20],
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=5,
    verbose=2,
)

# Fit grid search
grid_search.fit(df_train[predictors], df_train["Target"])

# %%
# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Extract the best model and preprocessor
lgbm_best_model = grid_search.best_estimator_.named_steps["lgbm"]
preprocessor = grid_search.best_estimator_.named_steps["preprocess"]

# Refit with early stopping using callbacks
from lightgbm import early_stopping

early_stopping_callback = early_stopping(stopping_rounds=10, verbose=True)

lgbm_best_model.fit(
    preprocessor.transform(df_train[predictors]), df_train["Target"],
    eval_set=[(preprocessor.transform(df_test[predictors]), df_test["Target"])],
    eval_metric="multi_logloss",
    callbacks=[early_stopping_callback]
)

# %%
# Generate predictions after refit
lgbm_predicted_probabilities = lgbm_best_model.predict_proba(preprocessor.transform(df_test[predictors]))
lgbm_predicted_classes = lgbm_best_model.predict(preprocessor.transform(df_test[predictors]))

# Add probabilities for each class
for i, class_label in enumerate(lgbm_best_model.classes_):
    df_test[f"lgbm_proba_class_{class_label}"] = lgbm_predicted_probabilities[:, i]

# Add predicted classes
df_test["lgbm_pred"] = lgbm_predicted_classes

# Evaluate LGBM
lgbm_prob_columns = [f"lgbm_proba_class_{class_label}" for class_label in lgbm_best_model.classes_]

lgbm_eval = evaluate_predictions(
    df_test,
    outcome_column="Target",
    prob_column=lgbm_prob_columns,  # List of probability columns
    preds_column="lgbm_pred",       # Column for predicted classes
    model_name="LGBM"
)
print(lgbm_eval)

# %%
# Inspect the results
print("Predicted probabilities and classes:")
print(df_test[["lgbm_pred"] + [f"lgbm_proba_class_{class_label}" for class_label in lgbm_best_model.classes_]].head())

# Evaluate the tuned model
print(
    "training log-loss:  {}".format(
        log_loss(df_train["Target"], lgbm_best_model.predict_proba(preprocessor.transform(df_train[predictors])))
    )
)

print(
    "testing log-loss:  {}".format(
        log_loss(df_test["Target"], lgbm_best_model.predict_proba(preprocessor.transform(df_test[predictors])))
    )
)

print(
    "training accuracy:  {}".format(
        accuracy_score(df_train["Target"], lgbm_best_model.predict(preprocessor.transform(df_train[predictors])))
    )
)

print(
    "testing accuracy:  {}".format(
        accuracy_score(df_test["Target"], lgbm_best_model.predict(preprocessor.transform(df_test[predictors])))
    )
)

print("Classification report (testing set):\n")
print(classification_report(df_test["Target"], lgbm_best_model.predict(preprocessor.transform(df_test[predictors]))))


# %% 

# GLM Model: Confusion Matrix
plot_confusion_matrix(
    df_test,
    outcome_column="Target",
    preds_column="glm_pred",  # Column for predicted classes
    model_name="GLM"
)

# LGBM Model: Confusion Matrix
plot_confusion_matrix(
    df_test,
    outcome_column="Target",
    preds_column="lgbm_pred",  # Column for predicted classes
    model_name="LGBM"
)






# %%
# Access the full pipeline and its steps
fitted_pipeline = grid_search.best_estimator_
fitted_preprocessor = fitted_pipeline.named_steps["preprocess"]
lgbm_model = fitted_pipeline.named_steps["lgbm"]

# Dynamically get feature names based on transformers
spline_output_features = fitted_preprocessor.named_transformers_["spline"].named_steps[
    "spline"
].get_feature_names_out(input_features=spline_features)

cat_output_features = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(
    input_features=categoricals
)

ord_output_features = ordinal  # CustomOrdinalEncoder doesn't expand features

# Combine all output feature names
all_feature_names = (
    list(spline_output_features)
    + list(remaining_numericals)
    + list(cat_output_features)
    + list(ord_output_features)
)

# Get feature importance DataFrame
lgbm_feature_importance = get_lgbm_feature_importance(
    lgbm_model=lgbm_model,
    fitted_preprocessor=fitted_preprocessor,
    remaining_numericals=remaining_numericals,
    spline_features=spline_features,
    categoricals=categoricals,
    ordinal=ordinal
)

# Display top 10 features
print("Top 10 Features for LGBM Model:")
print(lgbm_feature_importance.head(10))

# Plot top 10 features
plot_lgbm_top_features(lgbm_feature_importance, top_n=10)

# Plot PDP for top 5 features
plot_pdp_top_features(
    lgbm_model=lgbm_model,  # Use LGBMClassifier directly
    fitted_preprocessor=fitted_preprocessor,
    df_train=df_train,
    predictors=predictors,
    feature_importance=lgbm_feature_importance,
    all_feature_names=all_feature_names,  # Fixed
    target_class=0,
    top_n=5
)


# %%

# GLM Model Learning Curve
print("Plotting Learning Curve for GLM...")
fitted_preprocessor_glm = glm_best_model.named_steps["preprocess"]
X_train_transformed_glm = fitted_preprocessor_glm.transform(df_train[predictors])
y_train_glm = df_train["Target"]
glm_model = glm_best_model.named_steps["logistic"]
plot_learning_curve(glm_model, X_train_transformed_glm, y_train_glm, 
                    title="Learning Curve for GLM (Logistic Regression)", cv=5, scoring="accuracy")

# LGBM Model Learning Curve
print("Plotting Learning Curve for LGBM...")

# Retrieve pipeline and extract components
fitted_pipeline_lgbm = grid_search.best_estimator_  # Full pipeline
fitted_preprocessor_lgbm = fitted_pipeline_lgbm.named_steps["preprocess"]
lgbm_model = fitted_pipeline_lgbm.named_steps["lgbm"]

# Transform the data using the preprocessor
X_train_transformed_lgbm = fitted_preprocessor_lgbm.transform(df_train[predictors])
y_train_lgbm = df_train["Target"]

# Plot the learning curve
plot_learning_curve(lgbm_model, X_train_transformed_lgbm, y_train_lgbm, 
                    title="Learning Curve for LGBM", cv=5, scoring="accuracy")



# %%
import dalex as dx

# %% GLM Model Explainer
glm_explainer = dx.Explainer(
    glm_best_model,
    df_test[predictors], 
    df_test["Target"],    
    label="GLM Model"
)

# %% LGBM Model Explainer
# Retrieve the pipeline and preprocessor
fitted_pipeline_lgbm = grid_search.best_estimator_  # GridSearchCV's best pipeline
fitted_preprocessor = fitted_pipeline_lgbm.named_steps["preprocess"]

# Step 1: Transform input data using the preprocessor
X_test_transformed = fitted_preprocessor.transform(df_test[predictors])

# Step 2: Extract expanded feature names dynamically
spline_output_features = fitted_preprocessor.named_transformers_["spline"].named_steps[
    "spline"
].get_feature_names_out(input_features=spline_features)

cat_output_features = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(
    input_features=categoricals
)

ord_output_features = ordinal  # Ordinal features remain unchanged
expanded_feature_names = (
    list(spline_output_features) 
    + list(remaining_numericals) 
    + list(cat_output_features) 
    + ord_output_features
)

# Step 3: Convert transformed data into a labeled DataFrame
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=expanded_feature_names)

# Step 4: Create dalex Explainer for LGBM Model
lgbm_explainer = dx.Explainer(
    fitted_pipeline_lgbm.named_steps["lgbm"],  # Extract the LGBM model
    X_test_transformed_df,                     # Transformed data with feature names
    df_test["Target"],                         # Target variable
    label="LGBM Model"
)

# %% SHAP Values for First Observation (LGBM)
print("Computing SHAP values for the first observation...")

# Step 5: Compute SHAP values for the first observation
shap_explanation = lgbm_explainer.predict_parts(X_test_transformed_df.head(1), type="shap")

# Step 6: Plot SHAP values
print("SHAP Values Plot:")
shap_explanation.plot()

# %% GLM and LGBM Performance Comparison
print("Model Performance Comparison:")

glm_perf = glm_explainer.model_performance()
lgbm_perf = lgbm_explainer.model_performance()

glm_perf.plot(lgbm_perf, title="Model Performance Comparison")



# %%
# Model performance for GLM
glm_perf = glm_explainer.model_performance()
print("GLM Model Performance:")
glm_perf.result


# %%
# Model performance for LGBM
lgbm_perf = lgbm_explainer.model_performance()
print("LGBM Model Performance:")
lgbm_perf.result


# %%
# Plot comparison of both models
glm_perf.plot(lgbm_perf, title="Model Performance Comparison")



# %%
# Variable importance for GLM
glm_vi = glm_explainer.model_parts()
print("GLM Variable Importance:")
print(glm_vi.result)

# Variable importance for LGBM
lgbm_vi = lgbm_explainer.model_parts()
print("LGBM Variable Importance:")
print(lgbm_vi.result)


