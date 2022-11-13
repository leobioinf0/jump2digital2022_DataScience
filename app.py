# app.py

def app():
    """Perform Random Forest Classification.
    Print out Test f1-score (macro).
    Return predictions.csv and predictions.json"""
    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    train = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Jump2digital+2022/train.csv', sep=";")
    X = train.iloc[:,:-1]
    y = train.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2)
    clf = RandomForestClassifier(random_state=2,n_jobs=-1, n_estimators= 405)
    clf.fit(X_train, y_train)
    print('Test f1-score (macro): {}'.format(f1_score(y_test, clf.predict(X_test), average='macro').round(3)))
    
    test = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Jump2digital+2022/test.csv', sep=";")
    preds = clf.predict(test)
    pd.DataFrame(preds,columns=["final_status"]).to_csv("predictions.csv", index=False)
    pd.DataFrame(preds,columns=["target"]).to_json(path_or_buf="predictions.json",indent=2)

if __name__ == "__main__":
    app()