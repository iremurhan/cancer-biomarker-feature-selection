import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot
import seaborn
from sklearn.feature_selection import RFE
import lime.lime_tabular

importances = []
def process_datasets():
    """
    Process datasets from liver cancer folder and platforms folder.
    
    - Reads CSV files from liver_cancer_folder, converts patient status to binary, and normalizes gene expression data.
    - Reads Excel files from platforms_folder and stores them in platform_datasets dictionary.
    
    Returns:
    - liver_cancer_datasets: Dictionary containing processed liver cancer datasets.
    - platform_datasets: Dictionary containing datasets from different platforms.
    """
    for filename in os.listdir(liver_cancer_folder):
        if filename.endswith('.csv'):
            dataset_path = os.path.join(liver_cancer_folder, filename)
            df = pd.read_csv(dataset_path)

            # Convert patient status to binary
            df['type'] = df['type'].apply(lambda x: 1 if x == 'HCC' else 0)

            # Normalize gene expression data
            gene_cols = df.columns.drop(['type', 'samples'])  # exclude 'samples' column
            scaler = MinMaxScaler()
            df[gene_cols] = scaler.fit_transform(df[gene_cols])
            
            liver_cancer_datasets[filename] = df
    
    for filename in os.listdir(platforms_folder):
        if filename.endswith('.xlsx'):
            dataset_path = os.path.join(platforms_folder, filename)
            df = pd.read_excel(dataset_path)
            platform_datasets[filename] = df

    return liver_cancer_datasets, platform_datasets


def align_datasets_with_platforms():
    """
    Align datasets with their corresponding platforms.

    - Prints the files being processed.
    - Retrieves the CSV dataset and corresponding platform Excel dataset.
    - Creates a mapping from 'ID' to 'GCC' in the platform dataset.
    - Finds common features between the dataset and platform's IDs.
    - Drops columns from the dataset that are not in the platform's IDs.
    - Replaces features with 'GCC' values from the platform dataset.

    Returns:
    None
    """
    for csv_file, xlsx_file in liver_cancer_platforms.items():
        print(f'Processing {csv_file} with {xlsx_file}')  # Add a print statement to see which files are being processed

        # Get CSV dataset
        csv_df = liver_cancer_datasets[csv_file]

        # Get corresponding platform Excel dataset
        xlsx_df = platform_datasets[xlsx_file]

        # Create a mapping from 'ID' to 'GCC' in the platform dataset
        id_to_gcc = dict(zip(xlsx_df['ID'], xlsx_df['GB_ACC']))

        print(f'Finding mapped genes in {csv_file}')
        # Find common features between dataset and platform's IDs
        common_features = set(csv_df.columns).intersection(set(xlsx_df['ID']))

        print(f'Dropped unmapped genes in {csv_file}')
        # Drop columns from dataset that are not in platform's IDs
        columns_to_drop = [col for col in csv_df.columns if col not in common_features]
        csv_df.drop(columns=columns_to_drop, inplace=True)

        print(f'Replacing ID\'s with GB_ACC\'s in {csv_file}')
        # Replace features with 'GCC' value from platform dataset
        for feature in common_features:
            csv_df[feature] = csv_df[feature].map(id_to_gcc)

def evaluate_classifiers(X_train, y_train, X_test, y_test, classifiers):
    """
    Evaluate classifiers using the provided training and testing data.
    
    Args:
    - X_train: Training features.
    - y_train: Training target labels.
    - X_test: Testing features.
    - y_test: Testing target labels.
    - classifiers: Dictionary containing the classifiers to evaluate.
    
    Returns:
    - metrics: Dictionary containing evaluation metrics and predictions for each classifier.
    """
    # Additional storage for metrics and predictions
    metrics = {classifier: {} for classifier in classifiers}

    # Loop through classifiers
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        # Predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Confusion matrices
        train_conf_matrix = confusion_matrix(y_train, y_train_pred)
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        # Store metrics and predictions
        metrics[name]['train_accuracy'] = train_acc
        metrics[name]['test_accuracy'] = test_acc
        metrics[name]['train_confusion_matrix'] = train_conf_matrix
        metrics[name]['test_confusion_matrix'] = test_conf_matrix
        metrics[name]['train_predictions'] = y_train_pred
        metrics[name]['test_predictions'] = y_test_pred

        # Feature importances for Random Forest
        if name == "Random Forest":
            metrics[name]['feature_importances'] = clf.feature_importances_

    return metrics


def process_and_evaluate_cancer_datasets(classifiers, cancer_types, cancer_folders):
    """
    This function processes datasets for each cancer type, trains classifiers, extracts important features, 
    and identifies common important features across all datasets for each classifier.

    Args:
    - classifiers: A dictionary holding the classifiers.
    - cancer_types: A list of cancer types.
    - cancer_folders: A list of folders where datasets for each cancer type are stored.

    Returns: None
    """

    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import lime.lime_tabular

    # Loop through each cancer type
    for i in range(len(cancer_types)):

        print(f"Processing {cancer_types[i]} datasets")

        # Create list to store datasets
        datasets = []

        # Process datasets for current cancer type
        for filename in os.listdir(cancer_folders[i]):
            if filename.endswith('.csv'):
                dataset_path = os.path.join(cancer_folders[i], filename)
                df = pd.read_csv(dataset_path)
                datasets.append(df)

        # Store important features for each classifier
        important_features = {
            "Logistic Regression": [],
            "Random Forest": [],
        }

        # Loop through each dataset
        for df in datasets:

            # Remove the first column as it's not a feature
            df = df.drop(df.columns[0], axis=1)

            # Label encoding for the target classes
            le = LabelEncoder()
            df['type'] = le.fit_transform(df['type'])

            # Splitting into features (X) and target (y)
            X = df.drop('type', axis=1)
            y = df['type']

            # Apply PCA for dimensionality reduction
            n_components = min(X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components) # Adjust based on data
            X_pca = pca.fit_transform(X)

            # Splitting the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

            # Initialize the explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=[f'PC{i}' for i in range(X_train.shape[1])], class_names=le.classes_, discretize_continuous=True)

            # Loop through classifiers
            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)

                # Calculate LIME importances for a sample of instances
                n_samples = min(200, X_train.shape[0])  # Adjust based on your data
                sample_instances = X_train[np.random.choice(X_train.shape[0], n_samples, replace=False)]

                for instance in sample_instances:
                    exp = explainer.explain_instance(instance, clf.predict_proba, num_features=X_train.shape[1])
                    importances.append(dict(exp.as_list()))

                importance_df = pd.DataFrame(importances)

                # Get top 10 important features
                top_features = importance_df.mean().sort_values(ascending=False).head(10).index
                important_features[name].append(top_features)

        # Find and print common important features for each classifier
        for name, features in important_features.items():
            common_features = set(features[0]).intersection(*features)
            print(f"Common important features for {name} in {cancer_types[i]} datasets:")
            print(common_features)


def evaluate_classifier_importances(classifiers, cancer_types, cancer_folders):
    """
    This function processes and evaluates datasets for different cancer types. It calculates 
    the feature importances according to different classifiers and identifies common important 
    features across all datasets for each classifier.

    Args:
    - classifiers (dict): A dictionary of classifiers to use.
    - cancer_types (list): A list of strings representing the types of cancer.
    - cancer_folders (list): A list of directories where the datasets are stored.

    Returns:
    - None
    """
    # Loop through each cancer type
    for i in range(len(cancer_types)):

        print(f"Processing {cancer_types[i]} datasets")

        # Create list to store datasets
        datasets = []

        # Process datasets for current cancer type
        for filename in os.listdir(cancer_folders[i]):
            if filename.endswith('.csv'):
                dataset_path = os.path.join(cancer_folders[i], filename)
                df = pd.read_csv(dataset_path)
                datasets.append(df)

        # Store important features for each classifier
        important_features = {
            "Logistic Regression": [],
            "Random Forest": [],
        }

        # Loop through each dataset
        for df in datasets:

            # Remove the first column as it's not a feature
            df = df.drop(df.columns[0], axis=1)

            # Label encoding for the target classes
            le = LabelEncoder()
            df['type'] = le.fit_transform(df['type'])

            # Splitting into features (X) and target (y)
            X = df.drop('type', axis=1)
            y = df['type']

            # Apply PCA for dimensionality reduction
            n_components = min(X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components) # Adjust based on data
            X_pca = pca.fit_transform(X)

            # Splitting the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

            # Initialize the explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=[f'PC{i}' for i in range(X_train.shape[1])], class_names=le.classes_, discretize_continuous=True)

            # Loop through classifiers
            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)

                # Calculate LIME importances for a sample of instances
                n_samples = min(200, X_train.shape[0])  # Adjust based on your data
                sample_instances = X_train[np.random.choice(X_train.shape[0], n_samples, replace=False)]

                importances = []
                for instance in sample_instances:
                    exp = explainer.explain_instance(instance, clf.predict_proba, num_features=X_train.shape[1])
                    importances.append(dict(exp.as_list()))

                importance_df = pd.DataFrame(importances)

                # Get top 10 important features
                top_features = importance_df.mean().sort_values(ascending=False).head(10).index
                important_features[name].append(top_features)

        # Find and print common important features for each classifier
        for name, features in important_features.items():
            common_features = set(features[0]).intersection(*features)
            print(f"Common important features for {name} in {cancer_types[i]} datasets:")
            print(common_features)


from sklearn.feature_selection import RFE

def evaluate_classifier_rfe(classifiers, cancer_types, cancer_folders, num_features_to_select):
    """
    This function processes and evaluates datasets for different cancer types. It calculates 
    the feature importances according to different classifiers and identifies important features 
    using Recursive Feature Elimination (RFE).

    Args:
    - classifiers (dict): A dictionary of classifiers to use.
    - cancer_types (list): A list of strings representing the types of cancer.
    - cancer_folders (list): A list of directories where the datasets are stored.
    - num_features_to_select (int): The number of features to select.

    Returns:
    - None
    """
    # Loop through each cancer type
    for i in range(len(cancer_types)):

        print(f"Processing {cancer_types[i]} datasets")

        # Create list to store datasets
        datasets = []

        # Process datasets for current cancer type
        for filename in os.listdir(cancer_folders[i]):
            if filename.endswith('.csv'):
                dataset_path = os.path.join(cancer_folders[i], filename)
                df = pd.read_csv(dataset_path)
                datasets.append(df)

        # Store important features for each classifier
        important_features = {
            "Logistic Regression": [],
            "Random Forest": [],
        }

        # Loop through each dataset
        for df in datasets:

            # Remove the first column as it's not a feature
            df = df.drop(df.columns[0], axis=1)

            # Label encoding for the target classes
            le = LabelEncoder()
            df['type'] = le.fit_transform(df['type'])

            # Splitting into features (X) and target (y)
            X = df.drop('type', axis=1)
            y = df['type']

            # Splitting the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            # Loop through classifiers
            for name, clf in classifiers.items():
                # Fit the classifier
                clf.fit(X_train, y_train)

                # Apply RFE for feature selection
                selector = RFE(clf, n_features_to_select=num_features_to_select, step=1)
                selector = selector.fit(X_train, y_train)

                # Get the most important features
                important_features_mask = selector.support_
                important_features[name] = [feature for feature, selected in zip(X_train.columns, important_features_mask) if selected]

        # Print important features for each classifier
        for name, features in important_features.items():
            print(f"Important features for {name} in {cancer_types[i]} datasets:")
            print(features)


# Define folder paths
liver_cancer_folder = os.path.join('.', 'liver_cancer')
platforms_folder = os.path.join('.', 'platforms')

# Dictionary to store datasets for different platforms
platform_datasets = {}
# Dictionary to store liver cancer datasets
liver_cancer_datasets = {}

liver_cancer_platforms = {
    # GPL571	[HG-U133A_2] Affymetrix Human Genome U133A 2.0 Array
    # GPL3921	[HT_HG-U133A] Affymetrix HT Human Genome U133A Array
    'Liver_GSE14520_U133A.csv': 'GPL571-17391.xlsx',
    # GPL570	[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0 Array 
    'Liver_GSE62232.csv': 'GPL570-55999.xlsx'}

# Create a dictionary to hold the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(),
}
# Define cancer types
cancer_types = ['liver_cancer'] # , 'breast_cancer']
cancer_folders = [liver_cancer_folder] # , breast_cancer_folder]



liver_cancer_datasets, platform_datasets = process_datasets()
align_datasets_with_platforms()
process_and_evaluate_cancer_datasets(classifiers, cancer_types, cancer_folders)
evaluate_classifier_importances(classifiers, cancer_types, cancer_folders)
# evaluate_classifier_rfe(classifiers, cancer_types, cancer_folders, num_features_to_select=10)