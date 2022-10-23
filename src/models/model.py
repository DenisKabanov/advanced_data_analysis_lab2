import category_encoders as ce
import src.config as cfg
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor

real_pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
    ]
)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]
)

preprocess_pipe = ColumnTransformer(transformers=[
    ('real_cols', real_pipe, cfg.FLOAT_COLS + cfg.INT_COLS + cfg.NEW_COLS),
    ('cat_cols', cat_pipe, cfg.CAT_COLS),
    ]
)

model = CatBoostRegressor(iterations = 1000, learning_rate = 0.01, loss_function = 'MAPE', random_seed = 7)
