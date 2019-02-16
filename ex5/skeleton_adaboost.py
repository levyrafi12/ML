#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T, inverse_vocab):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n_samples = X_train.shape[0]
    D_t = np.ones(n_samples) / n_samples
    alpha_vals = []
    h_vals = []

    f = np.zeros(n_samples)

    for t in range(T):
        print("t = {}".format(t))
        h_pred, h_index, h_theta, eps_t = weak_learner_dt(D_t, X_train, y_train) # use decision tree

        print("h_pred = {}, word = {}, theta = {}, eps = {}".format(\
            h_pred, inverse_vocab[h_index], h_theta, eps_t))

        h_vals.append((h_pred, h_index, h_theta))
        alpha_vals.append(0.5 * np.log((1 - eps_t) / eps_t)) # hypothesis weight
        alpha_t = alpha_vals[-1]

        f += alpha_t * estimate_y(X_train, h_pred, h_index, h_theta)
 
        g = np.sign(f)

        print("train error = {}, alpha = {}".format(np.sum(y_train != g), alpha_t))
        # update D_t
        D_t_1 = D_t * np.exp(-alpha_t * y_train * estimate_y(X_train, h_pred, h_index, h_theta))
        D_t = D_t_1 / np.sum(D_t_1)
        
    return h_vals, alpha_vals

# You can add more methods here, if needed.

def weak_learner_dt(D, X_train, y_train):
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    best_eps = 1

    clf = DecisionTreeClassifier(random_state=0, max_depth=1)

    for j in range(n_features):
        clf.fit(X_train[:, j].reshape(-1,1), y_train)
        theta = clf.tree_.threshold[0]
        for sgn_val in [-1, 1]:
            eps = np.sum(D * (y_train != estimate_y(X_train, sgn_val, j, theta)))
            
            if eps < best_eps:
                best_theta = theta
                best_sgn_val = sgn_val
                best_ind = j
                best_eps = eps

    return best_sgn_val, best_ind, best_theta, best_eps

def weak_learner(D, X_train, y_train):
    n_features = X_train.shape[1]

    best_eps = 1

    for j in range(n_features):
        word_counts = np.unique(X_train[:, j].reshape(-1,1)) 
        for theta in word_counts:
            for sgn_val in [-1, 1]:
                eps = np.sum(D * (y_train != estimate_y(X_train, sgn_val, j, theta)))
                if eps < best_eps:
                    best_theta = theta
                    best_sgn_val = sgn_val
                    best_ind = j
                    best_eps = eps

    return best_sgn_val, best_ind, best_theta, best_eps

def estimate_y(X_train, h_pred, h_index, h_theta):
    return (2 * (X_train[:, h_index] <= h_theta) - 1) * h_pred

def calc_error(h_vals, alpha_vals, x_data, y_data):
    n_samples = x_data.shape[0]

    err_vals = []
    f = np.zeros(n_samples)

    for (h_pred, h_index, h_theta), alpha in zip(h_vals, alpha_vals):
        f += alpha * estimate_y(x_data, h_pred, h_index, h_theta)
        g = np.sign(f) # the classifier
        err_vals.append(np.average(y_data != g))

    return err_vals

def calc_loss(h_vals, alpha_vals, x_data, y_data):
    """
        compute exponential loss
    """
    n_samples = x_data.shape[0]
    loss_vals = []
    f = np.zeros(n_samples)
    for (h_pred, h_index, h_theta), alpha in zip(h_vals, alpha_vals):
        f += alpha * estimate_y(x_data, h_pred, h_index, h_theta)
        loss_vals.append(np.average(np.exp(-y_data * f)))
    return loss_vals

def gen_plots(h_vals, alpha_vals, X_train, y_train, X_test, y_test):
    train_error = calc_error(h_vals, alpha_vals, X_train, y_train)
    test_error = calc_error(h_vals, alpha_vals, X_test, y_test)

    plt.plot(train_error, label='training error')
    plt.plot(test_error, label='test error')
    plt.xlabel('epoch')
    plt.ylabel('error in percent')
    plt.legend(loc='upper right')
    plt.savefig('train_test_err')
    plt.show()
    plt.close()

    train_loss = calc_loss(h_vals, alpha_vals, X_train, y_train)
    test_loss = calc_loss(h_vals, alpha_vals, X_test, y_test)

    plt.plot(train_loss, label='training loss')
    plt.plot(test_loss, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('average exponential loss')
    plt.legend(loc='upper right')
    plt.savefig('train_test_loss')
    plt.show()

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, inverse_vocab) = data

    h_vals, alpha_vals = run_adaboost(X_train, y_train, 80, inverse_vocab)

    gen_plots(h_vals, alpha_vals, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()



