from PyALE import ale
def ale_plot(feature):
    df=pd.DataFrame(X_train_prep,columns=transformer.get_feature_names())
    ale_eff = ale(X=df, model=clf, feature=[feature])