"""
This file trains and evaluated a XGBoost classifier designed to predict rookie performance based
on combine and college career data.
"""

import pandas as pd
import nfl_data_py as nfl

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class RookieClassifier:
    def __init__(self, years, position):
        self.years = years
        self.position = position

        # Initialize DataFrames of rookie data and select features.
        self.rookie_data = self.get_rookie_data()
        self.features = ['draft_ovr', 'ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
        self.target = 'prod_level'

        # Initialize empty training and evaluation sets.
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Initialize training methods.
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Initialize the spot to store the best model from CV search.
        self.best_xgb_model = None

    def get_rookie_data(self):
        """
        Pulls and preprocesses combine and fantasy data for rookie players.
        """

        print(f"Fetching rookie data for {self.position}s from {self.years[0]} to {self.years[-1]}...")

        # Pull NFL combine data for the given year range and restrict to fantasy relevant positions and features.
        combine_df = nfl.import_combine_data(years=self.years, positions=[self.position])

        combine_features = ["season",  "draft_ovr", 'pfr_id', 'player_name', 'pos', 'ht', 'wt', 'forty',
        'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
        combine_df = combine_df[combine_features]
        combine_df.rename(columns={"season": "combine_year"}, inplace=True)
        combine_df.dropna(subset=["pfr_id"], inplace=True)

        # Pull fantasy point stats for the given year range
        stats_df = nfl.import_seasonal_data(years=self.years)[["player_id", "season", "fantasy_points_ppr"]]
        stats_df.dropna(subset=["player_id"], inplace=True)

        # Create a new DataFrame from stats_df giving only the first year stats of each player.
        first_season_stats_df = stats_df.sort_values(by=["player_id", "season"]).groupby("player_id").head(1)
        first_season_stats_df.rename(columns={"season": "rookie_season"}, inplace=True)

        # Pull the id mapping dataframe as combine_df and stats_df use different id's.
        id_df = nfl.import_ids()[["pfr_id", "gsis_id"]]
        id_df.rename(columns={"gsis_id": "player_id"}, inplace=True)  #  Match column value as in combine_df.

        # Add the prf_id column into first_season_stats.
        first_season_stats_df = first_season_stats_df.merge(id_df, how="left", on="player_id")

        # Combine the rookie season stats with the combine data into a single DataFrame.
        rookie_df = combine_df.merge(first_season_stats_df, how="left", on="pfr_id")

        # Filter out rookies who did not participate in their rookie season.
        rookie_df.dropna(subset=["rookie_season"], inplace=True)
        rookie_df.reset_index(drop=True, inplace=True)

        # Convert height string into an integer.
        rookie_df["ht"] = rookie_df["ht"].astype(str)
        for index, row in rookie_df.iterrows():
            if row["ht"] != "None":
                height = row["ht"].split("-")
                rookie_df.loc[index, "ht"] = int(height[0]) * 12 + int(height[1])
            else:
                rookie_df.loc[index, "ht"] = None
        rookie_df["ht"] = rookie_df["ht"].astype(float)

        # Compute quartiles of fantasy point totals.
        quartiles = rookie_df['fantasy_points_ppr'].quantile([1/3, 2/3])
        q1, q2 = quartiles[1/3], quartiles[2/3]

        # Label the production of each players rookie season by fantasy point quartiles.
        rookie_df["prod_level"] = ""

        for index, row in rookie_df.iterrows():
            if row['fantasy_points_ppr'] >= q2:
                rookie_df.loc[index, "prod_level"] = "Elite"
            elif row['fantasy_points_ppr'] >= q1:
                rookie_df.loc[index, "prod_level"] = "Average"
            else:
                rookie_df.loc[index, "prod_level"] = "Bust"

        print(f"Data loaded and preprocessed. Number of rookies: {len(rookie_df)}")

        return rookie_df

    def prepare_data(self):
        """Prepares the data for training."""

        # Create training features and target
        X = self.rookie_data[self.features]
        y = self.rookie_data[self.target]

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data into train and temp sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=13, stratify=y_encoded
        )

        # Fit scaler on training data and transform all sets
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test

        print(f"Data split complete: Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")

    def train_model(self):
        """Trains an XGBoost model with GridSearchCV."""

        # Define XGBoost model.
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', # Multi-class classification
            num_class=len(self.label_encoder.classes_), # Specify number of classes
            random_state=13,
            eval_metric='mlogloss', # Multi-class logloss
        )

        # Hyperparameter Tuning (GridSearchCV)
        param_grid = {
            'n_estimators': [200],  # Number of trees in the ensemble
            'max_depth': [3],  # Max depth of each tree
            'learning_rate': [0.05],
            'subsample': [0.8],  # Fraction of samples used for each tree
            'colsample_bytree': [0.7],  # Fraction of features used for each tree
            'gamma': [0],  # Min loss reduction required to perform split
            'reg_alpha': [0.01],  # L1 regularization
            'reg_lambda': [1]  # L2 regularization
        }

        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        print("GridSearchCV finished.")
        print(f"Best Score ({grid_search.scoring}): {grid_search.best_score_:.4f}")

        # Store the best model found by GridSearchCV
        self.best_xgb_model = grid_search.best_estimator_
        print("\nBest Hyperparameters Found by GridSearchCV:")
        print(grid_search.best_params_)

    def evaluate_model(self):
        """Evaluates the trained model on the test set."""

        print("\nEvaluating model on the test set...")

        # Model Evaluation
        y_pred = self.best_xgb_model.predict(self.X_test)

        # Decode labels for reporting
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)  # Decode predictions
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)  # Decode test labels

        print("\nModel Evaluation:")
        report = classification_report(y_test_decoded, y_pred_decoded, labels=self.label_encoder.classes_, zero_division=0)
        print(report)

        print("Confusion Matrix:")
        confusion = confusion_matrix(y_test_decoded, y_pred_decoded, labels=self.label_encoder.classes_)
        print(confusion)

        # Feature Importance
        print("\nFeature Importance:")
        importances = self.best_xgb_model.feature_importances_
        feature_importances = pd.DataFrame({'feature': self.features, 'importance': importances})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        print(feature_importances)

# Example usage:
years = list(range(2000, 2025))
for position in ["RB"]:
    classifier = RookieClassifier(years, position)
    classifier.prepare_data()
    classifier.train_model()
    classifier.evaluate_model()




