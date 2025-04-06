import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, auc, roc_curve
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

__import__('warnings').filterwarnings('ignore')

st.title("Student Depression Prediction")

# Data Visualization Part

st.markdown('# Data Visualization')

# Load the Data using Pandas
df = pd.read_csv('./data/Student Depression Dataset.csv')

# Exploratory Data Analysis for plotting the Graphs
numerical_vars = [col for col in df.select_dtypes(include=np.number).columns.tolist() if col not in ['Depression', 'id']]
categorical_vars = df.select_dtypes(exclude=np.number).columns.tolist()

# Plotting The Graphs

st.markdown('###  Select type of Analysis')
type = ['Univariate', 'Bivariate']
analysis = st.selectbox("Analysis Type", type)

# st.markdown('Continous or Categarical')
variable = ['Continous Variables' , 'Categorical Variables']
var_select = st.selectbox('Variable Type', variable)

if st.button("Generate Plots"):
    
    if analysis == 'Univariate':
        if var_select == 'Continous Variables':
            fig, axs = plt.subplots(nrows=len(numerical_vars), figsize=(10, 5 * len(numerical_vars)))
            for i, var in enumerate(numerical_vars):
                sns.histplot(df[var], kde=True, ax=axs[i])
                axs[i].set_title(f'Distribution of {var}')
            plt.tight_layout()
            st.pyplot(fig)

        elif var_select == 'Categorical Variables':
            for var in categorical_vars:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(x=df[var], ax=ax)
                ax.set_title(f'Distribution of {var}')
                plt.xticks(rotation=90)
                st.pyplot(fig)

    elif analysis == 'Bivariate':
        if var_select == 'Continous Variables':
            for var in numerical_vars:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(x=df['Depression'], y=df[var], ax=ax)
                ax.set_title(f'{var} by Depression')
                st.pyplot(fig)

        elif var_select == 'Categorical Variables':
            for var in categorical_vars:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(x=var, hue='Depression', data=df, ax=ax)
                ax.set_title(f'{var} distribution by Depression')
                plt.xticks(rotation=90)
                st.pyplot(fig)

# Confuction Matrix for every ML Algorithm
st.markdown('# Machine Learning Algorithm Performance')

algo = ['Logistice Regression (LogisticRegression)', 'Random Forest Classifier (RandomForestClassifier)', 'XGBoost Classifier (XGBClassifier)', 'Light Gradient Boosting Classifier (LGBMClassifier)']
algo_select = st.selectbox('Select The Machine Learning Algorithm' , algo)

# Function for Algo evaluation

for var in numerical_vars:
    df[var] = df[var].fillna(df[var].median())
for var in categorical_vars:
    df[var] = df[var].fillna(df[var].mode()[0])

for var in numerical_vars:

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[var], color='skyblue')
    plt.title(f'before treatment: {var}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = df[(df[var] < lower) | (df[var] > upper)]
    print(f"{var}: {len(outliers)} outliers detected")

    df[var] = np.where(df[var] < lower, lower, np.where(df[var] > upper, upper, df[var]))

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[var], color='lightgreen')
    plt.title(f"after treatment: {var}")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def process_city_column(df, 
                       anomaly_mapping=None, 
                       threshold=50, 
                       top_n=10,
                       other_label='Other'):

    if anomaly_mapping is None:
        anomaly_mapping = {
            'Less than 5 Kalyan': 'Kalyan',
            'Less Delhi': 'Delhi',
            'M.Com': other_label,
            'ME': other_label,
            '3.0': other_label
        }
    
    df = df.copy()
    df['City_original'] = df['City']
    
    df['City'] = df['City'].replace(anomaly_mapping)
    
    city_counts = df['City'].value_counts()
    
    replace_cities = city_counts[city_counts < threshold].index
    df['City'] = df['City'].replace(replace_cities, other_label)
    
    top_cities = df['City'][df['City'] != other_label].value_counts().nlargest(top_n).index
    df['City'] = np.where(df['City'].isin(top_cities), df['City'], other_label)
    
    city_order = [other_label] + list(top_cities)
    df['City'] = pd.Categorical(df['City'], categories=city_order, ordered=True)
    
    print("\nfinal distribution")
    print(df['City'].value_counts().sort_index(ascending=False))
    
    return df

df = process_city_column(df, threshold=50, top_n=10)

plt.figure(figsize=(12,6))
sns.countplot(y='City', data=df, order=df['City'].cat.categories)
plt.title('distribution after processing')
plt.show()

hierarchies = {
    'degree': [
        'Class 12',
        'B.Arch', 'B.Pharm', 'B.Tech', 'B.Com', 'BBA', 'BHM', 'B.Ed', 'BSc', 'BA', 'BCA', 'LLB', 'BE', 'MBBS',
        'M.Tech', 'MBA', 'MCA', 'MA', 'M.Com', 'M.Ed', 'ME', 'MHM', 'M.Pharm', 'MSc', 'LLM',
        'PhD', 'MD',
        'Others'
    ],
    'sleep': [
        'More than 8 hours',
        '7-8 hours',
        '5-6 hours',
        'Less than 5 hours',
        'Others'
    ],
    'dietary': ['Healthy', 'Moderate', 'Unhealthy', 'Others']
}

categorical_pipeline = Pipeline(steps=[
    ('encoder', ColumnTransformer(
        transformers=[
            ('binary', OrdinalEncoder(), [
                'Gender', 
                'Have you ever had suicidal thoughts ?', 
                'Family History of Mental Illness'
            ]),
            ('degree', OrdinalEncoder(categories=[hierarchies['degree']]), ['Degree']),
            ('dietary', OrdinalEncoder(categories=[hierarchies['dietary']]), ['Dietary Habits']),
            ('sleep', OrdinalEncoder(categories=[hierarchies['sleep']]), ['Sleep Duration']),
            ('categorical', OneHotEncoder(drop='first'), ['City', 'Profession'])
        ],
        remainder='drop'
    ))
])

numeric_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler())    
])

pipeline = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, categorical_vars),
        ('num', numeric_pipeline, numerical_vars)
    ],
    remainder='drop'
)

df_processed = pipeline.fit_transform(df)
X, y = df_processed, df['Depression']


from time import perf_counter
from functools import wraps

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {perf_counter() - start:.2f} seconds')
        return result
    return wrapper
@timeit
def train_model_and_evaluate(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show classification report
    report = classification_report(y_test, y_pred)
    st.markdown("### Classification Report")
    st.text(report)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    cm.plot(ax=ax, colorbar=False)
    plt.grid(False)
    st.markdown("### Confusion Matrix")
    st.pyplot(fig)

if st.button('Evaluate'):
    if algo_select == 'Logistice Regression (LogisticRegression)':
        train_model_and_evaluate(LogisticRegression())

    elif algo_select == 'Random Forest Classifier (RandomForestClassifier)':
        train_model_and_evaluate(RandomForestClassifier())

    elif algo_select == 'XGBoost Classifier (XGBClassifier)':
        train_model_and_evaluate(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

    elif algo_select == 'Light Gradient Boosting Classifier (LGBMClassifier)':
        train_model_and_evaluate(LGBMClassifier())


st.markdown('# ROC Curve')
if st.button("Plot ROC Curve"):

    models = {
        'LogisticRegression': LogisticRegression(),
        'LGBMClassifier': LGBMClassifier(verbosity=-1),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'RandomForestClassifier': RandomForestClassifier()
    }

    params = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        },
        'LGBMClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
            'max_depth': [-1, 5],
            'boosting_type': ['gbdt']
        },
        'XGBClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
            'objective': ['binary:logistic']
        },
        'RandomForestClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5],
        }
    }

    best_models = {}

    @timeit
    def run_grid_search():
        for model_name, model in models.items():
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=5,
                scoring='f1_macro',
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            st.write(f"✅ Best params for `{model_name}`:", grid_search.best_params_)

    run_grid_search()

    # Create VotingClassifier from best models
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    best_models['VotingClassifier'] = voting_clf

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, model in best_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} AUC: {auc_score:.2f}")

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')

    st.pyplot(fig)



# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input features
features = [
    'Gender', 'Age', 'Academic Pressure', 'CGPA', 'Sleep Duration', 'Dietary Habits',
    'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
    'Family History of Mental Illness', 'Financial Stress'
]

# Define categorical options
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Sleep Duration': ['Less than 4 hours', '4-6 hours', '6-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Unhealthy'],
    'Degree': ['Undergraduate', 'Graduate', 'Postgraduate'],
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Family History of Mental Illness': ['Yes', 'No']
}

st.markdown('# Machine Learning Model Prediction')

st.write("Enter the details below to predict depression risk:")

# User input fields
user_input = {}
for feature in features:
    if feature in categorical_options:
        user_input[feature] = st.selectbox(f"{feature}", categorical_options[feature])
    else:
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Convert input into DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    if prediction == 1:
        st.error("The model predicts a high risk of depression.")
    else:
        st.success("The model predicts a low risk of depression.")
