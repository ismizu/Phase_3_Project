def base_add_score(name, y_train, train_pred, y_test, test_pred):

    train_acc = round(accuracy_score(y_train, train_pred), 2)
    train_recall = round(recall_score(y_train, train_pred), 2)
    train_f1 = round(f1_score(y_train, train_pred), 2)

    test_acc = round(accuracy_score(y_test, test_pred), 2)
    test_recall = round(recall_score(y_test, test_pred), 2)
    test_f1 = round(f1_score(y_test, test_pred), 2)
    
    global log_imbalance_df
    
    log_imbalance_df = log_imbalance_df.append({'name': name,
                                                'accuracy_train': train_acc,
                                                'accuracy_test': test_acc,
                                                'recall_train': train_recall,
                                                'recall_test': test_recall,
                                                'f1_score_train': train_f1,
                                                'f1_score_test': test_f1},
                                               ignore_index = True)

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