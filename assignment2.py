def drop_fist_last(grads):
    first, *middle, last = grads
    return avg(middle)

grad = (1, 2, 3, 4, 5)
drop_first_last(grad)

#%matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10
 

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


#plt.figure()
#plt.scatter(X_train, y_train, label='training data')
#plt.scatter(X_test, y_test, label='test data')
#plt.legend(loc=4);

X_train = X_train.reshape(X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], 1)
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    answer = np.zeros((4, 100))
    degress = [1, 3, 6, 9]
    for index, d in enumerate(degress):
        featurizer = PolynomialFeatures(degree=d)
        X_poly = featurizer.fit_transform(X_train)
        clf = LinearRegression()
        clf.fit(X_poly, y_train)
        eval_data = np.linspace(0,10,100).reshape(100, 1)
        eval_poly = featurizer.fit_transform(eval_data)
        answer[index] = clf.predict(eval_poly)
    
    return answer# Return your answer


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    
    for d in range(10):
        featurizer = PolynomialFeatures(degree=d)
        X_poly = featurizer.fit_transform(X_train)
        clf = LinearRegression()
        clf.fit(X_poly, y_train)
        # train score
        r2_train[d] = clf.score(X_poly, y_train)
        # test score
        test_poly = featurizer.fit_transform(X_test)
        r2_test[d] = clf.score(test_poly, y_test)

    return (r2_train, r2_test)# Your answer here


def answer_three():
    
    # Your code here 
#    plt.figure(figsize=(10,5))
    r2_train, r2_test = answer_two()
#    plt.plot(range(10), r2_train, label='training data')
#    plt.plot(range(10), r2_test, label='test data')
    
    return (0, 9, 6)# Return your answer


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here
    featurizer = PolynomialFeatures(degree=12)
    X_poly = featurizer.fit_transform(X_train)
    test_poly = featurizer.fit_transform(X_test)
    
    # without regularization
    clf = LinearRegression()
    clf.fit(X_poly, y_train)
    # train score
    LinearRegression_R2_test_score = clf.score(test_poly, y_test)
    
    # Lasso
    from sklearn import linear_model
    reg = linear_model.Lasso(alpha = 0.01, max_iter=10000)
    reg.fit(X_poly, y_train)
    Lasso_R2_test_score = reg.score(test_poly, y_test)
    

    return(LinearRegression_R2_test_score, Lasso_R2_test_score) # Your answer here


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)
X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the following (X_train2, X_test2, y_train2, y_test2) for questions 5 through 7:
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    # Your code here
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train2, y_train2)
    importance = clf.feature_importances_
    index = np.argsort(-importance)
    answer = []
    for i in range(10):
        _ = index[i]
        answer.append(X_mush.columns[_])
    return(answer) # Your answer here
answer_five()


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    model = SVC(kernel='rbf', C=1)
    model.fit(X_train2,y_train2)
    train_score, test_score = validation_curve(model, X_test2, y_test2, 'gamma', np.logspace(-4, 1, 6))
    tran_answer = np.mean(train_score, axis=1)
    test_answer = np.mean(test_score, axis=1)
    return(tran_answer, test_answer)# Your answer here


def answer_seven():
    
    # Your code here
    train_scores, test_scores = answer_six()
#    plt.figure(figsize=(10,5))
#    plt.plot(np.logspace(-4, 1, 6), train_scores)
#    plt.plot(np.logspace(-4, 1, 6), test_scores)
    
    return (0.0001, 10, 0.01)# Return your answer