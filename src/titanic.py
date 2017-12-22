from sklearn import tree, preprocessing, ensemble, neural_network, model_selection, metrics

from analysis import *

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/test.csv')
data_all = pd.concat([data_train, data_test], ignore_index=True)

#print(data_all.info())

@curried
def label_encode_notnull(columns, data):
    encoders = { column: preprocessing.LabelEncoder() for column in columns }

    def encode(column):
        encoders[column.name].fit(column.dropna())
        return column.map(lambda value: encoders[column.name].transform([value]) if pd.notnull(value) else value)

    return pd.concat([data.drop(columns, axis='columns'), data[columns].apply(encode)], axis='columns')

@curried
def onehot_encode_notnull(columns, data):
    return pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=columns, sparse=False, drop_first=False)

def encode_age_group(df):
    return df.assign(AgeGroup = df['AgeGroup'].map(lambda interval: interval.left).astype(np.int32))

label_encode_features = label_encode_notnull(['Pclass', 'Sex', 'Embarked', 'Title', 'Sex_Class' ])
onehot_encode_features = onehot_encode_notnull(['Pclass', 'Sex', 'Embarked', 'Title', 'Sex_Class'])   #, 'AgeGroup'])
encode_features = compose(onehot_encode_features, encode_age_group)

prepare_data = compose(
                 encode_features,
                 drop_columns(['Name', 'FamilyName', 'Cabin', 'Ticket', 'PassengerId', 'Survived']),
                 fill_fare(outlier_low=2, outlier_high=60),
                 add_bounded_fare(outlier_low=2, outlier_high=60),
                 fill_embarked,
                 add_age_group,
                 fill_age,
                 add_sex_and_class,
                 add_group_size,
                 add_cabin_size,
                 add_ticket_size,
                 add_family_size,
                 add_family_name,
                 compose(normalize_title, add_title)
               )

survived = data_train['Survived']
data_all_fixed = prepare_data(data_all)

data_train_fixed = data_all_fixed[:len(data_train)]
data_test_fixed = data_all_fixed[len(data_train):]

#print(data_all_fixed.info())
#print(data_all_fixed.isnull().any())

@curried
def select_features(selector, features, labels):
    selector.fit(features, labels)
    return pd.Series(selector.scores_, index=features.columns).sort_values(ascending=False)

#score_func=lambda X, y: feature_selection.mutual_info_classif(X, y, discrete_features=[data_all_fixed.columns.get_loc(i) for i in discrete_columns])
score_func=lambda X, y: feature_selection.mutual_info_classif(X, y)
show_top_features = select_features(feature_selection.SelectPercentile(score_func=score_func, percentile=100))

@curried
def select_percentile(features, percentile=100):
    discrete = ['Pclass', 'Sex', 'Title', 'GroupSize', 'FamilySize', 'TicketSize', 'CabinSize', 'Parch', 'SibSp', 'Embarked', 'AgeGroup']
    discrete_pos = [features.columns.get_loc(i) for i in list(filter(lambda x: x in discrete, features.columns))]
    score_func = lambda X, y: feature_selection.mutual_info_classif(X, y, discrete_features=discrete_pos, random_state=0)

    return feature_selection.SelectPercentile(score_func=score_func, percentile=percentile)

@curried
def select_k_best(features, k='all'):
    discrete = ['Pclass', 'Sex', 'Title', 'GroupSize', 'FamilySize', 'TicketSize', 'CabinSize', 'Parch', 'SibSp', 'Embarked', 'AgeGroup']
    discrete_pos = [features.columns.get_loc(i) for i in list(filter(lambda x: x in discrete, features.columns))]
    score_func = lambda X, y: feature_selection.mutual_info_classif(X, y, discrete_features=discrete_pos, random_state=0)

    return feature_selection.SelectKBest(score_func=score_func, k=k)

def select_top_features(features, labels, k='all'):
    select = select_k_best(features, k)

    select.fit(features, labels)

    return pd.Series(select.scores_, index=features.columns).sort_values(ascending=False)

ada_boost = { 'classifier': lambda: ensemble.AdaBoostClassifier(random_state=0),
              'parameters': {
                  'classification__n_estimators': [100]
              }
            }
gradient_boost = { 'classifier': lambda: ensemble.GradientBoostingClassifier(random_state=0),
                   'parameters': {
                       'classification__min_samples_split': [i / 10.0 for i in range(1, 10)],
                       'classification__min_samples_leaf': [i / 10.0 for i in range(1, 5)]
                   }
                 }
random_forest = { 'classifier': lambda: ensemble.RandomForestClassifier(random_state=0),
                  'parameters': {
                      'classification__n_estimators': [100],
#                      'classification__class_weight': ['balanced'],
                      'classification__criterion': ['entropy', 'gini'],
                      'classification__max_features': [None, 'auto'],
                      'classification__min_samples_split': [i / 10.0 for i in range(1, 10)],
                      'classification__min_samples_leaf': [i / 10.0 for i in range(1, 5)]
                  }
                }
knn = { 'classifier': lambda: neighbors.KNeighborsClassifier(),
        'parameters': {
            'classification__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'classification__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            'classification__n_neighbors': [200, 100, 50, 10, 5]
        }
      }

decision_tree = { 'classifier': lambda: tree.DecisionTreeClassifier(random_state=0), 
                  'parameters': {
#                      'classification__class_weight': ['balanced'],
                      'classification__criterion': ['entropy', 'gini'],
                      'classification__max_features': [None, 'auto'],
                      'classification__min_samples_split': [i / 10.0 for i in range(1, 10)],
                      'classification__min_samples_leaf': [i / 10.0 for i in range(1, 5)]
                  }
                }
classifiers = [
    #Ensemble Methods
#    ada_boost,
#    gradient_boost,
#    lambda: ensemble.BaggingClassifier(),
#    lambda: ensemble.ExtraTreesClassifier(),
#    gradient_boost,
#    random_forest,
#
#    #Gaussian Processes
#    lambda: gaussian_process.GaussianProcessClassifier(),
#    
#    #GLM
#    lambda: linear_model.LogisticRegressionCV(),
#    lambda: linear_model.PassiveAggressiveClassifier(),
#    lambda: linear_model.RidgeClassifierCV(),
#    lambda: linear_model.SGDClassifier(),
#    lambda: linear_model.Perceptron(),
#    
#    #Navies Bayes
#    lambda: naive_bayes.BernoulliNB(),
#    lambda: naive_bayes.GaussianNB(),
#    
#    #Nearest Neighbor
#    knn,
#    
#    #SVM
#    lambda: svm.SVC(probability=True),
##    svm.NuSVC(probability=True),
#    lambda: svm.LinearSVC(),
#    
#    #Trees    
#    lambda: tree.ExtraTreeClassifier(),
    decision_tree
]

def evaluate2(classifier, params, features, labels):
    n_splits = 2
    n_repeats = 1
    cv = model_selection.RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2)

    model = pipeline.Pipeline([
#      ('feature_selection', select_k_best(features, 'all')),
      ('feature_selection', select_percentile(features, 100)),
      ('classification', classifier())
    ])

#    clf = model_selection.GridSearchCV(model, params, cv=cv, scoring=metrics.make_scorer(metrics.precision_score, pos_label=1))
    clf = model_selection.GridSearchCV(model, params, cv=cv, scoring='accuracy')
    clf.fit(features_train, labels_train)

    print('=========='*8)
    print(clf.best_estimator_)
    if hasattr(clf.best_estimator_.named_steps['classification'], 'feature_importances_'):
        scores = pd.DataFrame(
                    index=[ features.columns[i] for i in clf.best_estimator_.named_steps['feature_selection'].get_support(indices=True) ]
                  , data={'Importance': clf.best_estimator_.named_steps['classification'].feature_importances_}
                 )
        print(scores[scores['Importance'] > 0].sort_values(by=['Importance'], ascending=False))

    scoring = {
        'accuracy':  metrics.make_scorer(metrics.accuracy_score),
        'f1':        metrics.make_scorer(metrics.f1_score),
        'precision': metrics.make_scorer(metrics.precision_score),
        'recall':    metrics.make_scorer(metrics.recall_score)
    }
    scores = model_selection.cross_validate(clf.best_estimator_, features_test, labels_test, scoring=scoring, cv=cv)
    print({ score: np.mean(scores['test_{}'.format(score)]) for score in scoring.keys() })

#    predicted = model_selection.cross_val_predict(clf.best_estimator_, features_test, labels_test, cv=model_selection.StratifiedKFold(n_splits=10))
#    print(metrics.classification_report(labels_test, predicted))
#    print('---')
#    print(pd.DataFrame(metrics.confusion_matrix(labels_test, predicted, labels=[0,1])))

    return model


#    predictions = model.predict(data_test_fixed)
#    print(predictions)

#print(select_top_features(data_train_fixed, survived))

def evaluate(features, labels):
    n_splits = 2
    n_repeats = 5
    cv = model_selection.RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2)

    model = tree.DecisionTreeClassifier(random_state=0)
    params = {
        'class_weight': ['balanced'],
        'criterion': ['entropy', 'gini'],
        'max_features': [None, 'auto'],
        'min_samples_split': [i / 10.0 for i in range(1, 10)],
        'min_samples_leaf': [i / 10.0 for i in range(1, 5)]
    }

    clf = model_selection.GridSearchCV(model, params, cv=cv, scoring='accuracy')
    clf.fit(features_train, labels_train)

    scoring = {
        'accuracy':  metrics.make_scorer(metrics.accuracy_score),
        'f1':        metrics.make_scorer(metrics.f1_score),
        'precision': metrics.make_scorer(metrics.precision_score),
        'recall':    metrics.make_scorer(metrics.recall_score)
    }
    validation = model_selection.cross_validate(clf.best_estimator_, features_test, labels_test, scoring=scoring, cv=cv)
    scores = { score: [np.mean(validation['test_{}'.format(score)])] for score in scoring.keys() }

    return (clf.best_estimator_, scores)


for i in range(1,5):
    for classifier in classifiers:
        model, scores = evaluate(data_train_fixed, survived)

        importance = pd.DataFrame(index=[data_train_fixed.columns], data={'Importance': model.feature_importances_})

        print(importance[importance['Importance'] > 0].sort_values(by=['Importance'], ascending=False))
        print(pd.DataFrame.from_dict(scores, orient='index'))
