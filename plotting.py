import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import seaborn as sns 
import warnings
import shap
import alibi
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from alibi.explainers import ALE, plot_ale
import shap  # package used to calculate Shap values

df = pd.read_excel("data/data_project.xlsx")
df["MonthRefunding"] = df["CreditAmount"]/df["CreditDuration"]


#The replace function applies a dictionnary to the dataframe in order to 
#replace the codes (A70, A32 etc with understandable words (unemployed for 2 years, 
#has already a credit etc)

def replace(df):
    with open('config.json', "r") as f:
        
        data = f.read()
    # reconstructing the data as a dictionary
    js = json.loads(data)
    #On remplace les données codées par du texte
    df = df.replace(js)
    return df

df = replace(df)
le = OrdinalEncoder()
to_encode = ['CreditHistory', 'EmploymentDuration', 'Housing', 'Purpose', 'Savings']

for col in to_encode:
    df[col] = le.fit_transform(np.array(df[col]).reshape(-1, 1))
    
X, y = df.drop(["CreditRisk (y)", "y_hat"], axis=1), df["CreditRisk (y)"]
to_encode = ['CreditHistory', 'EmploymentDuration', 'Housing', 'Purpose', 'Savings']


#This function gets PDP values we will use after to compute our graphs

def get_PDPvalues(col_name, data, model, grid_resolution = 100):
    Xnew = data.copy()
    sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
    Y_pdp = []
    for each in sequence:
        Xnew[col_name] = each
        Y_temp = model.predict(Xnew)
        Y_pdp.append(np.mean(Y_temp))
    return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp})



#The EDA class is meant to display the first features analyses related to the target
#variable. The first function (density) shows the age distribution for each class (credit
#risked and not credit risked). The second function (dividing_class_plot) shows, for multiple 
#features (Employment duration, Credi duration, Credit amount) the difference between the two classes

class EDA:
    
    def density(self):
        
        # seaborn histogram
        fig, ax = plt.subplots(figsize=(12,8), dpi=300)

        sns.distplot(df[df["CreditRisk (y)"]==0]['Age'], hist=True, kde=True, 
                     bins=int(180/5), color = 'blue',
                     hist_kws={'edgecolor':'black'})
        # Add labels
        plt.title('Histogram of Age for risked credit people')
        plt.xlabel('Age(years)')
        plt.ylabel('Number of people');

        fig, ax = plt.subplots(figsize=(12,8), dpi=300)

        sns.distplot(df[df["CreditRisk (y)"]==1]['Age'], hist=True, kde=True, 
                     bins=int(180/5), color = 'blue',
                     hist_kws={'edgecolor':'black'})
        # Add labels
        plt.title('Histogram of Age for not-risked credit people')
        plt.xlabel('Age(years)')
        plt.ylabel('Number of people');
        
    def dividing_class_plot(self):
        
        fig, axes = plt.subplots(2, 2)

        df.groupby("EmploymentDuration")["CreditRisk (y)"].mean().sort_values(ascending=False).plot(kind="bar", figsize=(12, 12), title="Risk credit rate according to employment duration", ax=axes[1,1])
        df.groupby("CreditRisk (y)")["CreditDuration"].mean().plot(kind="bar", figsize=(12, 12), title="Risk credit rate according to credit duration", rot=0, ax=axes[0,1])
        df.groupby("CreditHistory")["CreditRisk (y)"].mean().sort_values(ascending=False).plot(kind="bar", figsize=(12, 13), title="Credit risk according to purpose product", ax=axes[1, 0])
        df.boxplot("CreditAmount", ax=axes[0,0], by="CreditRisk (y)");
     

#Here, we plot the interpretability graphs
    
class Interpretability:
    
    def __init__(self, model):
        
        self.m = model
        self.m.fit(X, y)
        
        
    def plotting_PDP(self):
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(18, 20))
        
        PDP_features = ["CreditDuration", "CreditAmount", "InstallmentRate", "Age", "NumberOfCredits", "MonthRefunding", "Gender", "Group"]
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        axes_twin = [str(i) + "_twin" for i in axes]
        
        for feature, ax in zip(PDP_features, axes):
            
            df = get_PDPvalues(feature, X, self.m)
            ax.plot(X[feature], np.zeros(X[feature].shape)+min(df['PDs'])-1, 'k|', ms=15)
            ax.plot(df[feature], df['PDs'], lw = 2)
            ax.set_ylabel(feature)
            
            ax_twin = ax.twinx()
            sns.distplot(X[feature], hist=False,
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'}, ax=ax_twin)
            ax.set_ylabel(feature)
            ax_twin.set_ylabel(feature + "  density",color="red",fontsize=14)
            ax.set_ylabel(feature)

            
    def ale_plot(self):
        
        try:
            
        
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(18, 20))

            ale = ALE(self.m.predict, feature_names=X.columns, target_names=["CreditRisk (y)"])
            ale_exp = ale.explain(np.array(X))

            ale_features = ["CreditDuration", "CreditAmount", "InstallmentRate", "Age", "NumberOfCredits", "MonthRefunding", "Gender", "Group"]
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

            for feature, axe in zip(ale_features, axes):
                plot_ale(ale_exp, features=[feature], ax=axe)
        
        except ValueError:
            
            self.m.fit(X.values, y)
            ale = ALE(self.m.predict, feature_names=X.columns, target_names=["CreditRisk (y)"])
            ale_exp = ale.explain(np.array(X.values))
            
            ale_features = ["CreditDuration", "CreditAmount", "InstallmentRate", "Age", "NumberOfCredits", "MonthRefunding", "Gender", "Group"]
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

            for feature, axe in zip(ale_features, axes):
                plot_ale(ale_exp, features=[feature], ax=axe)
    
    def shap_graph(self):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create object that can calculate shap values
        try:
            explainer = shap.TreeExplainer(self.m, X_train)

            # calculate shap values. This is what we will plot.
            # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
            shap_values = explainer.shap_values(X_test)

            # Make plot. Index of [1] is explained in text below.
            shap.summary_plot(shap_values, X_test)
        
        except Exception:
            
            explainer = shap.LinearExplainer(self.m, X_train)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)
        
        
        

