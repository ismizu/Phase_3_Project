#----- KNN -----#
knn_pipe = Pipeline([
        ('std_scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())])

kn_grid = {'knn__n_neighbors': list(range(1, 20, 2)),
           'knn__metric': ['minkowski', 'manhattan'],
           'knn__weights': ['uniform', 'distance']
}

gs = GridSearchCV(knn_pipe, param_grid = kn_grid, cv = 5, n_jobs = -1)#, scoring = 'f1')
gs.fit(X_train, y_train)

y_train_pred_knn = gs.best_estimator_.predict(X_train)
y_test_pred_knn = gs.best_estimator_.predict(X_test)
model_add_score('knn_', y_train, y_train_pred_knn, y_test, y_test_pred_knn)
gs.best_params_
#-- KNN end --#


#----- DT -----#
dtc_pipe = Pipeline([
        ('std_scaler', StandardScaler()),
        ('dtc', DecisionTreeClassifier(class_weight = 'balanced',
                                       random_state = 42))])

dtc_grid = {'dtc__max_depth': list(range(0, 40, 10)),
           'dtc__min_samples_leaf': list(range(0, 15, 5)),
           'dtc__max_leaf_nodes': list(range(100, 150, 10))
}

dtc_gs = GridSearchCV(dtc_pipe, param_grid = dtc_grid, cv = 5, n_jobs = -1)#, scoring = 'f1')
dtc_gs.fit(X_train, y_train)

y_train_pred_dtc = dtc_gs.best_estimator_.predict(X_train)
y_test_pred_dtc = dtc_gs.best_estimator_.predict(X_test)
model_add_score('dtc', y_train, y_train_pred_dtc, y_test, y_test_pred_dtc)
dtc_gs.best_params_
#-- DT end --#


#-----  RFR -----#
rfr_pipe = Pipeline([
        ('std_scaler', StandardScaler()),
        ('rfr', RandomForestClassifier(n_estimators = 200,
                                       class_weight = 'balanced'))])

rfr_grid = {'rfr__max_depth': list(range(30, 51, 10)),
           'rfr__min_samples_leaf': list(range(0, 10, 2)),
           'rfr__max_leaf_nodes': list(range(50, 151, 10)),
           'rfr__max_features': ['auto', .7, .8]}

rfr_gs = GridSearchCV(rfr_pipe, param_grid = rfr_grid, cv = 5, n_jobs = -1, verbose = 1)
rfr_gs.fit(X_train, y_train)

y_train_pred_rfr = rfr_gs.best_estimator_.predict(X_train)
y_test_pred_rfr = rfr_gs.best_estimator_.predict(X_test)
model_add_score('rfr_update', y_train, y_train_pred_rfr, y_test, y_test_pred_rfr)
rfr_gs.best_params_
#-- RFR end --#


#----- Adding scores to a df -----#
models_df = pd.DataFrame(columns = ['name',
                                    'accuracy_train', 'accuracy_test',
                                    'recall_train', 'recall_test',
                                    'f1_score_train', 'f1_score_test'])

def model_add_score(name, y_train, train_pred, y_test, test_pred):

    train_acc = round(accuracy_score(y_train, train_pred), 2)
    train_recall = round(recall_score(y_train, train_pred), 2)
    train_f1 = round(f1_score(y_train, train_pred), 2)

    test_acc = round(accuracy_score(y_test, test_pred), 2)
    test_recall = round(recall_score(y_test, test_pred), 2)
    test_f1 = round(f1_score(y_test, test_pred), 2)
    
    global models_df
    
    models_df = models_df.append({'name': name,
                                  'accuracy_train': train_acc,
                                  'accuracy_test': test_acc,
                                  'recall_train': train_recall,
                                  'recall_test': test_recall,
                                  'f1_score_train': train_f1,
                                  'f1_score_test': test_f1},
                                 ignore_index = True)