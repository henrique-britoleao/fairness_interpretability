from PyALE import ale


def ale_plot(feature):
    df = pd.DataFrame(X_train_prep, columns=transformer.get_feature_names())
    if feature in NUMERICAL_COLS:
        ale_eff = ale(X=df, model=clf, feature=[feature])
    else:
        n = transformer.transformers[0][2].index(feature)
        cols = df.columns.str.startswith("onehotencoder__x" + str(n))
        df = df.loc[:, ~cols]
        df = pd.concat(
            [
                df.reset_index().drop("index", axis=1),
                X_train[[feature]].reset_index().drop("index", axis=1),
            ],
            axis=1,
        )
        ohe = OneHotEncoder().fit(X_train[[feature]])
        print(df.info())

        def onehot_encode(feat):
            col_names = ohe.categories_[0]
            feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
            feat_coded.columns = col_names
            feat_coded.rename(
                lambda x: "onehotencoder__x" + str(n) + "_" + x, axis=1, inplace=True
            )
            return feat_coded.reset_index().drop("index", axis=1)

        ale_eff = ale(
            X=df,
            model=clf,
            feature=[feature],
            predictors=transformer.get_feature_names(),
            encode_fun=onehot_encode,
            include_CI=False,
        )
