import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class ztheta():
    def __init__(self, data, metadata, eta, num_steps, num_latents, alpha, plot=False, theta_update=True, theta_initial = None, C_initial = None):
        self.data = data
        self.metadata = metadata
        self.eta = eta
        self.num_steps = num_steps
        self.num_latents = num_latents
        self.theta_update = theta_update
        self.alpha = alpha
        self.converged=False

        self.z = np.random.default_rng().uniform(-0.2,0.2,(self.data.shape[0], self.num_latents))
        if theta_initial is None:
            self.theta = np.random.default_rng().uniform(-0.2,0.2,(self.num_latents,self.data.shape[1]))
            self.C = np.random.default_rng().uniform(-0.2, 0.2, (self.num_latents, self.metadata.shape[1]))
        else:
            self.theta = theta_initial
            self.C = C_initial

        self.loss_list, self.dcdl_list, self.dcdy_list, self.dcdC_list, self.z, self.theta, self.C = self.grad_decent(self.z, self.theta, self.C)
        self.grad_sum = self.dcdl_list + self.dcdy_list + self.dcdC_list

        if self.grad_sum[-1] <= 1e-2:
            self.converged=True
        if not self.theta_update and self.dcdl_list[-1] < 1e-2:
            self.converged=True

        self.theta -= self.theta.min(axis=1).reshape(-1,1)
        if plot:
            self.make_plots()

    def partition_func(self, l, y):
        return np.sum(np.exp(-np.dot(l, y)),axis=1)

    def q_calc(self, l,y):
        part_func = self.partition_func(l,y)
        return np.exp(-np.dot(l,y))/part_func[:,np.newaxis]

    def loss(self, l, y, C):
        z = self.partition_func(l,y)
        term_1 = np.sum(np.log(z))
        term_2 = np.sum(np.multiply(self.data,np.dot(l,y)))

        metadata_reconstruction = np.matmul(l, C)
        term_3 = norm(self.metadata - metadata_reconstruction)**2

        return self.alpha*(term_1 + term_2) + (1-self.alpha)*term_3

    def grad_decent_step(self, l, Y, C):
        '''
            l -> latents
            Y -> theta, consumer preference matrix
            C -> Metadata feature matrix
        '''
        pred = self.q_calc(l,Y)

        dcedl = np.dot(self.data,Y.T) - np.dot(pred,Y.T)
        dcedy = np.dot(l.T,self.data) - np.dot(l.T,pred)
        dcmdl = -2*np.matmul(self.metadata, C.T) + 2*np.matmul(np.matmul(l, C), C.T)
        dcmdC = -2*np.matmul(self.metadata.T, l).T + 2*np.matmul(l.T, np.matmul(l, C))

        dcdl = (1-self.alpha)*dcmdl + self.alpha*dcedl
        dcdC = (1-self.alpha)*dcmdC 
        dcdy = self.alpha*dcedy

        dcdlnorm = norm(dcdl)/norm(l)
        dcdynorm = norm(dcdy)/norm(Y)
        dcdCnorm = norm(dcdC)/norm(C)

        l = l - self.eta*dcdl
        if self.theta_update:
            Y = Y - self.eta*dcdy
            C = C - self.eta*dcdC
        # l[l>0] = 0
        # Y[Y<0] = 0
        return l, Y, C, dcdlnorm, dcdynorm, dcdCnorm

    def grad_decent(self, l, Y, C):
        loss_list = []
        
        dcdl_list = []
        dcdy_list = []
        dcdC_list = []

        l_grad = np.inf
        y_grad = np.inf
        C_grad = np.inf
        
        self.counter = 0
        while (l_grad + y_grad + C_grad > 1e-2) and self.counter <= self.num_steps:
            l, Y, C, dcdl, dcdy, dcdC = self.grad_decent_step(l,Y, C)
            dcdl_list.append(dcdl)
            dcdy_list.append(dcdy)
            dcdC_list.append(dcdC)
            loss_list.append(self.loss(l, Y, C))
            l_grad = dcdl
            y_grad = dcdy
            C_grad = dcdC
            self.counter += 1
            if not self.theta_update and l_grad < 1e-2:
                break

        return loss_list, dcdl_list, dcdy_list, dcdC_list, l, Y, C
    
    def make_plots(self):
        x = np.linspace(1,self.counter,self.counter)
        fig, axis = plt.subplots(1,2,figsize=(10,5))

        axis[0].plot(x,self.loss_list)
        axis[0].set_title('loss')
        axis[0].set_yscale('log')

        axis[1].plot(x,self.dcdl_list, label='dcdl', color='red')
        axis[1].plot(x,self.dcdy_list, label='dcdy', color='blue')
        axis[1].plot(x,self.dcdC_list, label='dcdC', color='green')
        axis[1].plot(x, np.array(self.dcdl_list) + np.array(self.dcdy_list) + np.array(self.dcdC_list), '--', label='gradient sum', color='black')
        axis[1].set_title('normalized gradient')
        axis[1].set_yscale('log')

        plt.legend()
        plt.show()
