import numpy as np

class HeartAttackPredictor:

    def __init__(self, learning_rate, x_train, x_test, y_train, y_test):
        ''' self initializaiton function '''

        self.lr = learning_rate
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.w = np.random.randn(x_train[1])
        self.b = np.random.randn(1)
        self.train_loss = []
        self.test_loss = []

    def sigmoid(self, x):
        ''' Sigmoid function '''
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, x):
        ''' forward function '''
        return self.sigmoid(np.dot(x, self.w) + self.b)

    def sigmoid_derivative(self, x):
        ''' deacivation function '''
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def backward_pass(self, x, y_groundtruths):
        ''' calculate gradients '''

        # calculate dL/dP
        y_predictions = self.forward_pass(x)
        dl_dp = 2 * (y_predictions - y_groundtruths)

        # calculate dP/dH
        hidden_state_values = np.dot(x, self.w) + self.b
        dp_dh = self.sigmoid_derivative(hidden_state_values)

        # calculate dH/dB and dH/dW
        dh_db = 1
        dh_dw = x


        # calculate dL/dW and dL/dB using chain rule
        dl_db = dh_db * dp_dh * dl_dp
        dl_dw = dh_dw * dp_dh * dl_dp

        return dl_db, dl_dw

    def optimizer(self, dl_db, dl_dw):
        ''' optimizer '''
        self.b = self.b + dl_db * self.lr
        self.w = self.w + dl_dw * self.lr
