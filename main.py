import numpy as np


def load_preprocess_data(kind='train'):
    """Load MNIST data from `path`"""
    if kind == 'train':
        digit_path = 'in_train.txt'
        labels_path = 'out_train.txt'
    else:
        digit_path = 'in_test.txt'
        labels_path = 'out_test.txt'

    digit = np.loadtxt(digit_path)/100
    labels = np.loadtxt(labels_path)[:, None]

    return digit, labels


class NeuralNetMLP(object):
    def __init__(self,
                 epochs=30, alpha=0.0):
        self.epochs = epochs
        self.alpha = alpha
        self.W = []
        self.b = []
        self.af = []
        self.layer_number = 0
        # confusion matrix
        self.cfm_c = []
        self.cfm_p = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def add_weights(self, inn, outn):
        self.W.append(-1 + 2*np.random.rand(outn, inn))
        return 0

    def add_bias(self, n):
        self.b.append(-1 + 2*np.random.rand(n,1))
        return 0

    def add_layer(self, in_size, out_size, activation_function):

        self.add_weights(in_size, out_size)
        self.add_bias(out_size)
        self.af.append(str(activation_function))
        return 0

    def _feedforward(self, X, W, b, af):
        """Compute feedforward step"""
        a = [X.reshape(16, 1)]
        self.layer_number = len(W)
        for i in range(self.layer_number):
            if af[i] == "sigmoid":
                a.append(self.sigmoid(np.dot(W[i], a[i]) + b[i]))
            elif af[i] == "softmax":
                a.append(self.softmax(np.dot(W[i], a[i]) + b[i]))
            else:
                print("no this activation function")
        return a

    def _backpropagate_the_error(self, a, t):
        L = self.layer_number
        error = [None] * L

        for i in reversed(range(L)):
            if i == self.layer_number - 1:
                F = np.diag(np.dot(np.diag(a[L].reshape(len(a[L]))), (1 - a[L])).reshape(len(a[L])))
                error[i] = -2 * np.dot(F, (t - a[L]))
            else:
                F = np.diag(np.dot(np.diag(a[i+1].reshape(len(a[i+1]))), (1 - a[i+1])).reshape(len(a[i+1])))
                error[i] = np.dot(F, (np.dot(self.W[i+1].transpose(), error[i+1]) ))

        return error


    def fit(self, X, y, print_progress=False):

        for eph in range(self.epochs):
            count = 0
            for i in range(X.shape[0]):

                t = np.zeros((10, 1))
                t[(int)(y[i])] = 1

                a = self._feedforward(X[i], self.W, self.b, self.af)
                error = self._backpropagate_the_error(a, t)

                delta_W = [0] * len(error)
                delta_b = error
                for j in range(len(error)):
                    delta_W[j] = np.dot(error[j], a[j].transpose())

                for j in range(len(delta_W)):
                    self.W[j] = self.W[j] - self.alpha * delta_W[j]
                for j in range(len(delta_b)):
                    self.b[j] = self.b[j] - self.alpha * delta_b[j]


                if np.argmax(a[self.layer_number]) == y[i]:
                    count += 1
            print('epho ', eph + 1, ' Train Correct Rate : ', count/X.shape[0])
        print("")


    def predict(self, X, y):
        count = 0
        for i in range(X.shape[0]):

            t = np.zeros((10, 1))
            t[(int)(y[i])] = 1

            a = self._feedforward(X[i], self.W, self.b, self.af)

            if np.argmax(a[self.layer_number]) == y[i]:
                count += 1
            else:
                # record for confusion matrix
                self.cfm_c.append(int(y[i]))
                self.cfm_p.append(np.argmax(a[self.layer_number]))
        print('Test Correct Rate : ', count / X.shape[0])


    def print_confusion_matrix(self):
        cf = np.zeros((10, 10))
        for i in range(10):
            for j in range(len(self.cfm_p)):
                if self.cfm_c[j] == i:
                    # print(self.cfm_c[j], " : ", self.cfm_p[j])
                    cf[self.cfm_p[j]][self.cfm_c[j]] += 1
        print(cf)




# load mnist data
X_train, y_train = load_preprocess_data('train')
X_test, y_test = load_preprocess_data('test')


nn = NeuralNetMLP(epochs=5,
                  alpha=0.25)

nn.add_layer(16, 100, "sigmoid")
nn.add_layer(100, 50, "sigmoid")
nn.add_layer(50, 25, "sigmoid")
nn.add_layer(25, 10, "softmax")


nn.fit(X_train, y_train, print_progress=True)

nn.predict(X_test, y_test)
nn.print_confusion_matrix()