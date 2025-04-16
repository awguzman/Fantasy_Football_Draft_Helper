"""
This file trains and evaluated a XGBoost classifier designed to predict player fantasy production using the previous season's stats.
It then uses the trained model to predict next years players to aid in drafting.
"""

import pandas as pd
import nfl_data_py as nfl
from scipy.stats import uniform, randint, loguniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class DraftTierClassifier:
    def __init__(self, years, position):
        self.years = years
        self.position = position

        # Features and target
        self.features = None
        self.training_features = None
        self.target = 'draft_tier'

        # Training, evaluation and prediction sets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_pred = None

        # Initialize necessary model methods.
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Initialize the spot to store the best model from CV search.
        self.best_xgb_model = None

        # Store player data.
        self.player_stats = None
        self.prev_season_stats = None

    def get_player_stats(self):
        print(f"Fetching player stats for {self.position}'s from {self.years[0]} to {self.years[-1]}...")

        # Select relevant features for the specified position.
        if self.position == "QB":
            self.features = ['player_id', 'season', 'completions', 'attempts', 'passing_yards', 'passing_tds',
                             'interceptions', 'sacks', 'passing_epa', 'fantasy_points_ppr']
        elif self.position == "RB":
            self.features = ['player_id', 'season', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'targets',
                             'receiving_yards', 'receiving_tds', 'fantasy_points_ppr']
        else:
            self.features = ['player_id', 'season', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'racr',
                             'wopr_x', 'dom', 'receiving_epa', 'fantasy_points_ppr']

        # Pull the id mapping dataframe for player information.
        id_df = nfl.import_ids()[['gsis_id', 'position', 'name', 'birthdate']]
        id_df.rename(columns={"gsis_id": "player_id"}, inplace=True)  #  Match column value as in combine_df.
        id_df.dropna(subset=['player_id'], inplace=True)

        # Pull fantasy point stats for the given year range restricting to position and features.
        stats_df = nfl.import_seasonal_data(years=self.years, s_type='REG')
        stats_df = stats_df.merge(id_df, how='right', on='player_id')
        stats_df = stats_df[stats_df['position'] == self.position]
        stats_df = stats_df[['position', 'name', 'birthdate'] + self.features]

        # Feature engineering: Compute player age.
        stats_df['age'] = stats_df['season'] - pd.to_datetime(stats_df['birthdate']).dt.year

        # Create a dataframe of only last year's stats.
        prev_season_df = stats_df[stats_df['season'] == self.years[-1]]

        # Add the next season fantasy point totals for each player which will serve as our training target.
        next_season_df = stats_df[['player_id', 'season', 'fantasy_points_ppr']].copy()
        next_season_df['season'] = next_season_df['season'] - 1  # Align season column for merging.
        next_season_df.rename(columns={'fantasy_points_ppr': 'next_season_fantasy_points_ppr'}, inplace=True)
        stats_df = pd.merge(stats_df, next_season_df, on=['player_id', 'season'], how='left')
        stats_df.dropna(subset=['next_season_fantasy_points_ppr'], inplace=True)  # Drop players without next year fantasy points.
        stats_df.reset_index(drop=True, inplace=True)  # Reset index after dropping.

        # Construct draft tiers based on next season fantasy points distributions.
        stats_df['draft_tier'] = pd.qcut(stats_df['next_season_fantasy_points_ppr'], q=5,
                                         labels=["Tier 5", "Tier 4", "Tier 3", "Tier 2", "Tier 1"]
                                         )

        # Feature cleaning.
        if self.position == "QB":
            self.features = ['player_id', 'name', 'season', 'age', 'completions', 'attempts', 'passing_yards',
                             'passing_tds', 'interceptions', 'sacks', 'passing_epa', 'fantasy_points_ppr', 'draft_tier']
        elif self.position == "RB":
            self.features = ['player_id', 'name', 'season', 'age', 'carries', 'rushing_yards', 'rushing_tds',
                             'rushing_epa', 'targets', 'receiving_yards', 'receiving_tds', 'fantasy_points_ppr', 'draft_tier']
        else:
            self.features = ['player_id', 'name', 'season', 'age', 'receptions', 'targets', 'receiving_yards',
                             'receiving_tds', 'racr', 'wopr_x', 'dom', 'receiving_epa', 'fantasy_points_ppr', 'draft_tier']
        stats_df = stats_df[self.features]

        print(f"Data loaded and preprocessed. Number of players: {len(stats_df)}")

        self.player_stats = stats_df
        self.prev_season_stats = prev_season_df

    def prepare_data(self):
        """Prepares and splits the data for training and evaluation."""

        if self.position == "QB":
            self.training_features = ['age', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
                                        'sacks', 'passing_epa']
        elif self.position == "RB":
            self.training_features = ['age', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_epa', 'targets',
                                      'receiving_yards', 'receiving_tds']
        else:
            self.training_features = ['age', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'racr',
                                      'wopr_x', 'dom', 'receiving_epa']

        # Create training features and target
        X = self.player_stats[self.training_features]
        y = self.player_stats[self.target]

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data into train and temp sets
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=13, stratify=y_encoded
        )

        # Fit scaler on training data and transform all sets.
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        print(f"Data split complete: Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")

    def train_model(self):
        """Trains a XGBoost classification model with RandomizedSearchCV."""

        # Define XGBoost model.
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', # Multi-class classification
            num_class=len(self.label_encoder.classes_), # Specify number of classes
            random_state=13,
            eval_metric='mlogloss', # Multi-class logloss
        )

        # Hyperparameter Tuning grid.
        param_dist = {
            'n_estimators': randint(100, 501),       # Number of trees
            'max_depth': randint(3, 7),             # Max depth of trees
            'learning_rate': loguniform(0.005, 0.2),

            'subsample': uniform(0.7, 0.3),          # Fraction of samples per tree
            'colsample_bytree': uniform(0.7, 0.3),   # Fraction of features per tree

            'gamma': uniform(0, .4),           # Min loss reduction for split
            'reg_alpha': loguniform(0.001, 1.0),           # L1 regularization
            'reg_lambda': loguniform(0.1, 10),         # L2 regularization
        }

        # Setup GridSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=200,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        print(f"Starting RandomizedSearchCV with {random_search.n_iter} iterations...")
        random_search.fit(self.X_train, self.y_train)

        print("RandomizedSearchCV finished.")
        print(f"Best Score ({random_search.scoring}): {random_search.best_score_:.4f}")

        # Store the best model found
        self.best_xgb_model = random_search.best_estimator_
        print("\nBest Hyperparameters Found by RandomizedSearchCV:")
        print(random_search.best_params_)

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
        feature_importances = pd.DataFrame({'feature': self.training_features, 'importance': importances})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        print(feature_importances)

    def predict_tiers(self):
        """Use the trained model to predict the draft tiers for last season's players."""

        # Prepare last seasons data for prediction.
        X_pred = self.prev_season_stats[self.training_features]
        self.X_pred = self.scaler.transform(X_pred)

        y_pred = self.best_xgb_model.predict(self.X_pred)
        pred_tiers = self.label_encoder.inverse_transform(y_pred)

        predict_df = self.prev_season_stats.copy()
        predict_df['predicted_draft_tier'] = pred_tiers

        print(predict_df[['name', 'season', 'predicted_draft_tier']])

years = list(range(2000, 2025))
positions = ["QB", "RB", "WR", "TE"]

for position in ["RB"]:
    classifier = DraftTierClassifier(years, position)
    classifier.get_player_stats()
    classifier.prepare_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.predict_tiers()
