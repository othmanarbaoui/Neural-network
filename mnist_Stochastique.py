import numpy as np
import time


class DNN :
    def __init__(self,sizes=[784 , 128 , 64 , 10] , epochs=10 , lr=0.001 , alpha=0.9,epsilon=1e-8):
        self.sizes = sizes 
        self.epochs=epochs
        self.lr = lr
        self.alpha=alpha
        self.epsilon = epsilon
        
        self.input_layer = sizes[0]
        self.hidden_1 = sizes[1]
        self.hidden_2 = sizes[2]
        self.output_layer = sizes[3]
        
        self.params={
            'W1' : np.random.randn(self.hidden_1 , self.input_layer) * np.sqrt(1./self.hidden_1),
            'W2' : np.random.randn(self.hidden_2 , self.hidden_1) * np.sqrt(1./self.hidden_2),
            'W3' : np.random.randn(self.output_layer , self.hidden_2) * np.sqrt(1./self.output_layer),
            'b1' : np.random.randn(self.hidden_1 ) * np.sqrt(1./self.hidden_1),
            'b2' : np.random.randn(self.hidden_2 ) * np.sqrt(1./self.hidden_2),
            'b3' : np.random.randn(self.output_layer ) * np.sqrt(1./self.output_layer)
        }
        self.params_2 = {
            'W1': np.zeros((self.hidden_1, self.input_layer)),
            'W2': np.zeros((self.hidden_2, self.hidden_1)),
            'W3': np.zeros((self.output_layer, self.hidden_2)),
            'b1': np.zeros((self.hidden_1)),
            'b2': np.zeros((self.hidden_2)),
            'b3': np.zeros((self.output_layer)),
        }
        
    
    def sigmoid(self , x , derivative = False) : 
        if derivative :
            return self.sigmoid(x , False)*(1 - self.sigmoid(x , False))
        else :
             return 1./(1 + np.exp(-x)) 
        

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        softmax_vals = exps / np.sum(exps, axis=0)
        if derivative:
            return softmax_vals * (1 - softmax_vals)
        else:
            return softmax_vals
    
        
    def forward_pass(self, x_train) :
        params = self.params
        params['A0'] = x_train

        params['Z1'] = np.dot(params['W1'] , params['A0'])+params['b1']
        params['A1'] = self.sigmoid(params['Z1'])

        params['Z2'] = np.dot(params['W2'] , params['A1'])+params['b2']
        params['A2'] = self.sigmoid(params['Z2'])

        params['Z3'] = np.dot(params['W3'] , params['A2'])+params['b3']
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']
    
    def backward_pass(self, y_train, output):
        params = self.params
        chang_w = {}

        dA3 = output - y_train  
        dZ3 = dA3 * self.softmax(params['Z3'], derivative=True)
        chang_w['W3'] = np.outer(dZ3, params['A2'])
        chang_w['b3'] = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(params['W3'].T, dZ3)
        dZ2 = dA2 * self.sigmoid(params['Z2'], derivative=True)
        chang_w['W2'] = np.outer(dZ2, params['A1'])
        chang_w['b2'] = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(params['W2'].T, dZ2)
        dZ1 = dA1 * self.sigmoid(params['Z1'], derivative=True)
        chang_w['W1'] = np.outer(dZ1, params['A0'])
        chang_w['b1'] = np.sum(dZ1, axis=0, keepdims=True)
        return chang_w
    
    def update_weights(self , change_w) : 
        for key , val in change_w.items() :
            self.params[key] -= self.lr  * val


    def update_weights_momentum(self, change_w):
        for key, val in change_w.items():
            self.params_2[key] = self.alpha * self.params_2[key] + (1 - self.alpha) * val    
            self.params[key] -= self.lr * self.params_2[key]


    def update_weights_adagrad(self, change_w):
        for key, val in change_w.items():
            self.params_2[key] += val ** 2
            self.params[key] -= self.lr / (np.sqrt(self.params_2[key] + self.epsilon)) * val        


    def update_weights_rmsprop(self, change_w):
        for key, val in change_w.items():
            self.params_2[key] = self.alpha * self.params_2[key] + (1 - self.alpha) * val**2
            self.params[key] -= self.lr / (np.sqrt(self.params_2[key]+ self.epsilon)) * val
    
    def compute_accuracy(self , test_Data) :
        prediction = []
        for x in test_Data : 
            values = x.split(",")
            input_data = (np.asfarray(values[1:])/255.0 * 0.99)+0.01
            targets =  np.zeros(10) +0.01
            targets[int(values[0])] = 0.99
            output = self.forward_pass(input_data)
            pred =  np.argmax(output)
            prediction.append(pred == np.argmax(targets))
        return np.mean(prediction)

    def train(self , train_list , test_list , update_method) : 
        start_time =  time.time()
        for i in range(self.epochs):
            np.random.shuffle(train_list)
            for x in train_list : 
                values = x.split(",")
                input_data = (np.asfarray(values[1:])/255.0 * 0.99)+0.01                
                targets =  np.zeros(10) +0.01
                targets[int(values[0])] = 0.99                    
                output = self.forward_pass(input_data)
                change_w = self.backward_pass(targets , output)
                match update_method.lower():
                    case 'standard':
                        self.update_weights(change_w)
                    case 'momentum':
                        self.update_weights_momentum(change_w)
                    case 'adagrad':
                        self.update_weights_adagrad(change_w)
                    case 'rmsprop':
                        self.update_weights_rmsprop(change_w)
                    case _:
                        self.update_weights(change_w)
            accuracy =  self.compute_accuracy(test_list)
            print('Epoch : {} , time spent : {:.02f}s , Accuracy : {:.2f}%'.format(i+1 , time.time()-start_time , accuracy*100))
    

if __name__ == "__main__":
    train_file = open("mnist_train.csv" , "r")
    train_list = train_file.readlines()
    train_file.close()
    test_file = open("mnist_test.csv" , "r")
    test_list = test_file.readlines()
    test_file.close()
    dnn = DNN(sizes=[784 , 128 , 64 , 10] , epochs=10 , lr=0.001)
    print("l entrainment a demarre")
    dnn.train(train_list , test_list , 'adagrad')