import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import learning_curve


import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df, figsize=(20, 10)):
    """
    Plots a heatmap representing the correlation matrix for all numeric columns in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    figsize : tuple, optional (default=(20, 10))
        The size of the figure to be plotted.

    Returns:
    --------
    None
        Displays a heatmap of the correlation matrix.

    Notes:
    ------
    - The heatmap uses the 'coolwarm' color map.
    - Numeric columns are automatically selected for correlation calculation.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()


def plot_categorical_distributions(df, figsize=(10, 4)):
    """
    Plots bar charts to visualize the distribution of values for each categorical column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    figsize : tuple, optional (default=(10, 4))
        The size of the figure for each categorical column plot.

    Returns:
    --------
    None
        Displays a series of bar plots for each categorical column.

    Notes:
    ------
    - Only columns with the 'object' data type are considered categorical.
    - Each plot visualizes the count of unique values for a single categorical column.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        plt.figure(figsize=figsize)
        sns.countplot(x=column, data=df)
        plt.ylabel('Count')
        plt.title(f'Distribution of {column}')
        plt.show()


def plot_pairplot(df, features, hue='Target'):
    """
    Generates a pairplot for selected features, colored by a specified target variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    features : list
        A list of column names (strings) to include in the pairplot.
    hue : str, optional (default='Target')
        The column name to use for color coding the points in the plot.

    Returns:
    --------
    None
        Displays a Seaborn pairplot for the specified features.

    Notes:
    ------
    - The `hue` parameter must reference a column in the DataFrame that can differentiate points.
    - Pairplots are useful for visualizing pairwise relationships between numerical variables.
    """
    sns.pairplot(df[features], hue=hue)
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_boxplot(df, figsize=(20, 10)):
    """
    Plots boxplots for all numerical columns in the provided DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing numerical data.
    figsize : tuple, optional (default=(20, 10))
        The size of the figure for the boxplot.

    Returns:
    --------
    None
        Displays a boxplot for numerical columns.

    Notes:
    ------
    - The function excludes columns that are not numerical.
    - It rotates x-axis labels for better readability when there are many columns.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    plt.figure(figsize=figsize)
    sns.boxplot(data=df[numerical_cols], orient='v', width=0.5)
    plt.title('Box Plot of Numerical Columns', fontsize=16)
    plt.xlabel('Numerical Columns', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histograms(df, figsize=(25, 20), color='green', edgecolor='black'):
    """
    Plots histograms for all columns in the provided DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    figsize : tuple, optional (default=(25, 20))
        The size of the figure for the histograms.
    color : str, optional (default='green')
        The color for the bars in the histogram.
    edgecolor : str, optional (default='black')
        The edge color for the bars in the histogram.

    Returns:
    --------
    None
        Displays a grid of histograms for all columns in the DataFrame.

    Notes:
    ------
    - Suitable for visualizing the distribution of numerical data.
    - Automatically adjusts spacing to avoid overlapping plots.
    """
    df.hist(figsize=figsize, color=color, edgecolor=edgecolor)
    plt.gcf().set_facecolor('white')
    plt.tight_layout()
    plt.show()


def plot_dist(dataset, columns_list, rows, cols):
    """
    Plots the distribution of specified numerical columns using KDE plots in a grid layout.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input DataFrame containing the data.
    columns_list : list
        A list of column names to include in the KDE plots.
    rows : int
        The number of rows in the grid layout.
    cols : int
        The number of columns in the grid layout.

    Returns:
    --------
    None
        Displays a grid of KDE plots for the specified numerical columns.

    Notes:
    ------
    - Columns named 'id' will be excluded from the plots.
    - Uses a custom color palette for better visual appeal.
    - Highlights the median value of each column with a dashed line.
    """
    from scipy.stats import skew

    columns_list = [col for col in columns_list if col != 'id']  # Exclude 'id' column if present
    fig, axs = plt.subplots(rows, cols, figsize=(60, 70))
    fig.subplots_adjust(hspace=1, wspace=1)  # Adjust spacing between subplots

    # Custom color palette
    colors = [
        '#E63946', '#F4A261', '#2A9D8F', '#264653', '#A8DADC',
        '#457B9D', '#1D3557', '#E9C46A', '#F77F00', '#D62828',
        '#003049', '#6A4C93', '#4CC9F0', '#7209B7', '#3A0CA3',
        '#F94144', '#F3722C', '#F8961E', '#90BE6D', '#43AA8B',
        '#577590', '#A44A3F', '#BB9457', '#6A994E', '#BC4749',
        '#C1121F', '#780000', '#3C096C', '#5A189A', '#9D4EDD',
        '#FF0054', '#FF7849', '#219EBC', '#FFB703', '#023047'
    ]

    # Plot KDE for each column
    for i, col in enumerate(columns_list):
        sns.kdeplot(
            dataset[col],
            ax=axs[i // cols, i % cols],
            fill=True,
            alpha=0.5,
            linewidth=0.5,
            color=colors[i % len(colors)],
            label='Density'
        )
        axs[i // cols, i % cols].set_title(f'{col}, Skewness: {skew(dataset[col]):.2f}', fontsize=28)
        axs[i // cols, i % cols].set_xlabel(col, fontsize=24)
        axs[i // cols, i % cols].set_ylabel('Density', fontsize=24)
        axs[i // cols, i % cols].legend(fontsize=22)
        axs[i // cols, i % cols].tick_params(axis='both', which='major', labelsize=20)

        # Highlight median value
        median_train = dataset[col].median()
        axs[i // cols, i % cols].axvline(x=median_train, color='#4caba4', linestyle='--')

    fig.suptitle('Distribution of Numeric Columns', fontsize=36)
    plt.tight_layout()
    plt.gcf().set_facecolor('skyblue')
    sns.despine(left=True, bottom=True)  # Remove borders for a cleaner look
    plt.show()

def plot_top_correlations(dataset, target_column, top_n=10, figsize=(10, 11)):
    """
    Plot the top N features with the highest correlation to the target column.

    Parameters:
    - data (DataFrame): The dataset containing the features and target column.
    - target_column (str): The name of the target column to analyze correlations with.
    - top_n (int): The number of top features to display (default is 10).
    - figsize (tuple): The size of the plot (default is (10, 11)).

    Returns:
    - None: Displays the bar plot of correlations.
    """
    # Calculate correlations
    correlations = dataset.corr()[target_column]
    
    # Get the top N features with highest absolute correlation
    top_features = correlations.abs().nlargest(top_n).index
    top_corr_values = correlations[top_features]
    
    # Plot the correlations
    plt.figure(figsize=figsize)
    plt.bar(top_features, top_corr_values,color='skyblue')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel(f'Correlation with {target_column}', fontsize=12)
    plt.title(f'Top {top_n} Features with Highest Correlation to {target_column}', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    plt.show()


def analyze_correlations(data, target_column, high_corr_threshold=0.7, low_corr_threshold=0.3):
    """
    Analyze correlations in the dataset, finding features highly and less correlated with the target variable 
    and plotting the correlation with the target variable.

    Parameters:
    - data (DataFrame): The dataset containing features and target column.
    - target_column (str): The name of the target column to analyze correlations with.
    - high_corr_threshold (float): The threshold for identifying highly correlated features (default is 0.7).
    - low_corr_threshold (float): The threshold for identifying less correlated features (default is 0.3).

    Returns:
    - highly_correlated (list): Features highly correlated with the target variable.
    - less_correlated (list): Features less correlated with the target variable.
    """
    correlation_with_target = data.corr()[target_column]
    
    # Identify highly correlated and less correlated features with the target
    highly_correlated = correlation_with_target[correlation_with_target.abs() >= high_corr_threshold].index.tolist()
    less_correlated = correlation_with_target[correlation_with_target.abs() <= low_corr_threshold].index.tolist()

    # Remove the target column itself from the lists
    if target_column in highly_correlated:
        highly_correlated.remove(target_column)
    if target_column in less_correlated:
        less_correlated.remove(target_column)

    # Print correlation with the target variable
    print("Correlation with Target:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(correlation_with_target.sort_values(ascending=False))

    # Plot correlation with the target variable
    plt.figure(figsize=(10, 8))
    correlation_with_target.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Correlation with Target Variable', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return highly_correlated, less_correlated


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, classification_report
from sklearn.model_selection import learning_curve
from sklearn.inspection import PartialDependenceDisplay

def plot_confusion_matrix(df, outcome_column, preds_column, model_name):
    """
    Plot a confusion matrix for classification models.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing actual and predicted values.
    outcome_column : str
        Column name for actual target values.
    preds_column : str
        Column name for predicted class labels.
    model_name : str
        Name of the model being evaluated (used for the plot title).

    Returns:
    --------
    None
        Displays the confusion matrix as a heatmap.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(df[outcome_column], df[preds_column])
    
    # Display confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", 
                xticklabels=np.unique(df[outcome_column]), 
                yticklabels=np.unique(df[outcome_column]))
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


def lorenz_curve(y_true, y_pred, weights=None):
    """
    Generate Lorenz curve values for calculating the Gini coefficient.

    Parameters:
    -----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    weights : array-like, optional
        Weights for each sample. If None, equal weights are assumed.

    Returns:
    --------
    cum_exposure : numpy.ndarray
        Cumulative proportion of observations.
    cum_true : numpy.ndarray
        Cumulative proportion of the true target values.

    Notes:
    ------
    The Lorenz curve can be used to compute the Gini index, which is a measure
    of inequality or model performance in certain domains.
    """
    order = np.argsort(y_pred)
    y_true, y_pred = y_true[order], y_pred[order]
    if weights is not None:
        weights = weights[order]
    else:
        weights = np.ones_like(y_true)
    cum_true = np.cumsum(y_true * weights) / np.sum(y_true * weights)
    cum_exposure = np.cumsum(weights) / np.sum(weights)
    return cum_exposure, cum_true


def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring="accuracy"):
    """
    Plot the learning curve for a given estimator.

    Parameters:
    -----------
    estimator : estimator object
        The machine learning model or pipeline to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    title : str, optional (default="Learning Curve")
        Title of the learning curve plot.
    cv : int, optional (default=5)
        Number of cross-validation folds.
    scoring : str, optional (default="accuracy")
        Scoring metric to evaluate the model performance.

    Returns:
    --------
    None
        Displays the learning curve with training and validation performance.
    """
    train_sizes = np.linspace(0.1, 1.0, 5)  # Varying training sizes
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, random_state=42
    )
    
    # Compute mean and standard deviation of scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, "o-", label="Training Score")
    plt.plot(train_sizes_abs, test_mean, "o-", label="Cross-Validation Score")
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def plot_pdp_top_features(lgbm_model, 
                          fitted_preprocessor, 
                          df_train, 
                          predictors, 
                          feature_importance, 
                          all_feature_names, 
                          target_class=0, 
                          top_n=5):
    """
    Plots Partial Dependence Plots (PDP) for the top N important features.

    Parameters:
    -----------
    lgbm_model : estimator
        Trained LightGBM model.
    fitted_preprocessor : sklearn ColumnTransformer
        Preprocessor pipeline fitted to training data.
    df_train : pd.DataFrame
        Training data containing predictors.
    predictors : list
        List of predictor column names.
    feature_importance : pd.DataFrame
        DataFrame containing 'Feature' and 'Importance' columns.
    all_feature_names : list
        List of all transformed feature names.
    target_class : int, optional (default=0)
        Target class index for multi-class models.
    top_n : int, optional (default=5)
        Number of top features to plot.

    Returns:
    --------
    None
        Displays PDP plots for the top N features.
    """
    top_features = feature_importance["Feature"].head(top_n).values
    print(f"Top {top_n} Features for PDP:", top_features)

    X_train_transformed = fitted_preprocessor.transform(df_train[predictors])

    # Generate PDP plots
    fig, ax = plt.subplots(figsize=(10, 8))
    PartialDependenceDisplay.from_estimator(
        lgbm_model,
        X_train_transformed,
        features=top_features,
        feature_names=all_feature_names,
        target=target_class,
        grid_resolution=50,
        ax=ax
    )

    plt.suptitle(f"Partial Dependence Plots for Top {top_n} Features (Class {target_class})", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_lgbm_top_features(feature_importance_df, top_n=10):
    """
    Plots the top N features by importance for an LGBM model.

    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame containing 'Feature' and 'Importance' columns.
    top_n : int, optional (default=10)
        Number of top features to display.

    Returns:
    --------
    None
        Displays a horizontal bar plot of the top N features by importance.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["Feature"][:top_n],
        feature_importance_df["Importance"][:top_n]
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances - LGBM")
    plt.xlabel("Importance Score")
    plt.show()
