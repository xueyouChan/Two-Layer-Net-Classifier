# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:42:08 2023

@author: Lenovo
"""

import sys
import random
import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt
sys.path.append('C:/Users/Lenovo/研一下/深度学习')  
class TwoLayerNet(object):
    def __init__(self,input_size,hidden_size,output_size,weight_init_std,types,lr=1e-3,reg=0.1):
        self.lr=lr
        self.reg=reg
        self.types=types
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        self.result={
                'h1':None,
                'relu':None,
                'h2':None,
                'out':None
            }
        self.gradient={
                'W2':None,
                'b2':None,
                'W1':None,
                'b1':None
            }

    def forward(self,X):
        self.result['h1']=np.dot(X,self.params['W1']) + self.params['b1']
        self.result['relu']=np.maximum(0,self.result['h1'])
        self.result['h2']=np.dot(self.result['relu'],self.params['W2'])+self.params['b2']     
        self.result['out']=self.softmax(self.result['h2'])
        return self.result['out']
    def softmax(self,h):
        exp_h = np.exp(h - np.max(h, axis=-1, keepdims=True))
        sm = exp_h/np.sum(exp_h, axis=-1, keepdims=True)
        return sm
    def CrossEntropy(self,out,y,epsilon=1e-12):
        out = np.clip(out, epsilon, 1.-epsilon)
        info=-np.log(out)
        return np.mean(y*info)
    def one_hot(self,y):
        encode=np.eye(self.types)[y]
        return encode
    def backpropagation(self,X,out,y,penalty='l2'):
        batch_num=X.shape[0]
        dL1 = (out - y)/batch_num
        self.gradient['W2']=np.dot(self.result['relu'].T,dL1)
        self.gradient['b2']=np.sum(dL1,axis=0)
        
        dL2=np.dot(self.params['W2'],dL1.T)
        dL2[np.dot(X, self.params['W1']+self.params['b1']).T <= 0] = 0 
        self.gradient['W1']=np.dot(dL2,X).T
        self.gradient['b1']=np.sum(dL2.T,axis=0)
        if penalty=='l2':
            self.gradient['W1'] += self.reg*self.params['W1']
            self.gradient['W2'] += self.reg*self.params['W2']
    def step(self):
        for item in self.params.keys():
            self.params[item]-=self.lr*self.gradient[item]
    def save_model(self,path):
        to_save={
            'model_name':'TwoLayerNet',
            'input_size':self.params['W1'].shape[0],
            'hidden_size':self.params['W1'].shape[1],
            'output_size':self.params['W2'].shape[1],
            'learning_rate':self.lr,
            'L2_rate':self.reg,
            'params':self.params
        }
        np.save(path,to_save)
    def load_model(self,path):
        to_fit=np.load(path)
        self.lr=to_fit['learning_rate']
        self.reg=to_fit['L2_rate']
        for item in self.params.keys():
            self.params[item]=to_fit['params'][item]
            
def train_test(epoch,batch_size,lr,net,lr_decay_rate=0.99,task='para_find'):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    x_train=x_train
    x_test=x_test
    idx=list(range(x_train.shape[0]))
    random.shuffle(idx)
    train_loss,train_acc=[],[]
    test_loss,test_acc=[],[]
    for e in range(epoch):
        loss_e=[]
        acc_e=[]
        for i in range(0,x_train.shape[0],batch_size) :
            X=x_train[idx[i:i+batch_size]]
            y=net.one_hot(t_train[idx[i:i+batch_size]])
            out=net.forward(X)
            loss_e.append(net.CrossEntropy(out,y))
            acc_e.append(calc_acc(out,y))
            net.backpropagation(X,out,y,'l2')
            net.step()
        if e//20 == 0 and e > 0:
            lr*=lr_decay_rate 
        
        train_loss.append(np.mean(loss_e))
        train_acc.append(np.mean(acc_e))

        test_out=net.forward(x_test)
        test_y=net.one_hot(t_test)
        test_loss.append(net.CrossEntropy(test_out,test_y))
        test_acc.append(calc_acc(test_out,test_y))
        print(f'---------epoch {e+1}---------')
        print(f'train loss {train_loss[-1]}')
        print(f'test loss {test_loss[-1]}')
        print(f'train acc {train_acc[-1]}')
        print(f'test acc {test_acc[-1]}')
    
    if task == 'train':
        draw(train_loss,test_loss,train_acc,test_acc,net.params)
        
    elif task == 'para_find':
        return train_loss, train_acc, test_loss, test_acc, net.params
        net.save_model("C:/Users/Lenovo/Desktop/资料/研一下/深度学习/HW1/model.npy")
    
def calc_acc(out,y):
    labels=np.argmax(y,axis=1)
    preds=np.argmax(out,axis=1)
    acc=(sum(labels==preds))/len(labels)
    return acc

def para_find(lr_cand, hidden_cand, reg_cand):
    test_acc = float('-inf')
    best_para = {'lr':None, 'hidden':None, 'reg':None}
    
    for lr in lr_cand:
        for hidden in hidden_cand:
            for reg in reg_cand:
                print(f'Testing the accuracy with lr {lr}, hidden {hidden}, reg {reg}')
                net = TwoLayerNet(784, hidden, 10, 0.01, 10, lr, reg)
                _, _, test, _, _= train_test(100, 256, lr, net)
                if test[-1] > test_acc:
                    test_acc = test[-1]
                    best_para['lr'] = lr
                    best_para['hidden'] = hidden
                    best_para['reg'] = reg
    return best_para['lr'], best_para['hidden'], best_para['reg']
                
                

def draw(train_loss,test_loss,train_acc,test_acc,params):
    W1, W2 = [], []
    for i in range(params['W1'].shape[0]):
        for j in range(params['W1'].shape[1]):
            W1.append(params['W1'][i][j])
    for i in range(params['W2'].shape[0]):
        for j in range(params['W2'].shape[1]):
            W2.append(params['W2'][i][j])
            
    x_list=np.arange(len(train_loss))
    train_loss = [float(item) for item in train_loss]
    test_loss = [float(item) for item in test_loss]
    train_acc = [float(item) for item in train_acc]
    test_acc = [float(item) for item in test_acc]
    
    plt.figure(figsize=(14,14),dpi=100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.subplot(2,2,1)
    plt.hist(W1, bins = 20, color = 'gold', edgecolor = 'b')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Values of W1')
    
    plt.subplot(2,2,2)
    plt.hist(W2, bins = 20, color = 'gold', edgecolor = 'b')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Value of W2')
    
    plt.subplot(2,2,3)
    plt.plot(x_list, train_loss, color="gold", linewidth=5.0, linestyle="-", label="train loss")
    plt.plot(x_list, test_loss, color="red", linewidth=5.0, linestyle="-", label="test loss")
    plt.legend(["train loss", "test loss"], ncol=2)
    plt.xlabel("epoch",fontsize = 20)
    plt.ylabel("Loss",fontsize = 20)
    plt.title('Training loss and testing loss')

    plt.subplot(2,2,4)
    plt.plot(x_list, train_acc ,color="gold", linewidth=5.0, linestyle="-", label="train acc")
    plt.plot(x_list, test_acc, color="red", linewidth=5.0, linestyle="-", label="test acc")
    plt.legend(["train acc", "test acc"], ncol=2)
    plt.xlabel("epoch",fontsize = 20)
    plt.ylabel("ACC",fontsize = 20)
    plt.title('Training accuracy and testing accuracy')
    plt.savefig('./plot.jpg',dpi=100)
            
if __name__ == '__main__':
    
    # 参数备选
    lr_cand = [0.01,0.005,0.001]
    hidden_cand = [100, 200, 300]
    reg_cand = [0.01,0.005,0.001]
    
    # 参数选择并保存模型
    
    lr, hidden, reg = para_find(lr_cand, hidden_cand, reg_cand)
    
    # 用经过参数查找后的模型进行测试
    net=TwoLayerNet()
    net.load_model('C:/Users/Lenovo/Desktop/资料/研一下/深度学习/HW1/model.npy')
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    y_test=net.one_hot(t_test)
    out=net.forward(x_test)
    print(calc_acc(out, y_test)) # 输出分类精度
    
    # 画图
    net=TwoLayerNet(784,300,10,0.01,10,lr = 0.005,reg = 0.001)
    train_test(100, 128, 0.005, net, task = 'train')
