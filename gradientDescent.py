'''
   Copyright 2024 Karthik Srinivasan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class ztheta():
    def __init__(self, data, eta, num_steps, num_latents, plot=False, theta_update=True, theta_initial = None):
        self.data = data
        self.eta = eta
        self.num_steps = num_steps
        self.num_latents = num_latents
        self.theta_update = theta_update

        self.z = -np.random.default_rng().uniform(0.2,0.5,(self.data.shape[0], self.num_latents))
        if theta_initial is None:
            self.theta = np.random.default_rng().uniform(0.2,0.5,(self.num_latents,self.data.shape[1]))
        else:
            self.theta = theta_initial

        self.loss_list, self.dcdl_list, self.dcdy_list, self.z, self.theta = self.grad_decent(self.z, self.theta)
        # self.make_plots()
        self.theta -= self.theta.min(axis=1).reshape(-1,1)
        if plot:
            self.make_plots()
    
    def partition_func(self, l, y):
        return np.sum(np.exp(-np.dot(l, y)),axis=1)

    def q_calc(self, l,y):
        part_func = self.partition_func(l,y)
        return np.exp(-np.dot(l,y))/part_func[:,np.newaxis]

    def loss(self, l, y):
        z = self.partition_func(l,y)
        term_1 = np.sum(np.log(z))
        term_2 = np.sum(np.multiply(self.data,np.dot(l,y)))
        return term_1 + term_2

    def grad_decent_step(self, l, Y):
        pred = self.q_calc(l,Y)
        dcdl = np.dot(self.data,Y.T) - np.dot(pred,Y.T)
        dcdy = np.dot(l.T,self.data) - np.dot(l.T,pred)
        dcdlnorm = norm(dcdl)/norm(l)
        dcdynorm = norm(dcdy)/norm(Y)
        l = l - self.eta*dcdl
        Y = Y - self.eta*dcdy
        # l[l>0] = 0
        # Y[Y<0] = 0
        return l, Y, dcdlnorm, dcdynorm
    
    def grad_decent_step_wo_theta(self, l, Y):
        pred = self.q_calc(l,Y)
        dcdl = np.dot(self.data,Y.T) - np.dot(pred,Y.T)
        dcdy = np.dot(l.T,self.data) - np.dot(l.T,pred)
        dcdlnorm = norm(dcdl)/norm(l)
        dcdynorm = norm(dcdy)/norm(Y)
        l = l - self.eta*dcdl
        # l[l>0] = 0
        # Y[Y<0] = 0
        return l, Y, dcdlnorm, dcdynorm

    def grad_decent(self, l, Y):
        loss_list = []
        dcdl_list = []
        dcdy_list = []
        l_grad = np.inf
        y_grad = np.inf
        self.counter = 0
        

        if self.theta_update:
            while (l_grad + y_grad > 1e-2) and self.counter <= self.num_steps:
                l, Y, dcdl, dcdy = self.grad_decent_step(l,Y)
                dcdl_list.append(dcdl)
                dcdy_list.append(dcdy)
                loss_list.append(self.loss(l, Y))
                l_grad = dcdl
                y_grad = dcdy
                self.counter += 1
        else:
            while (l_grad > 1e-2) and self.counter <= self.num_steps:
                l, Y, dcdl, dcdy = self.grad_decent_step_wo_theta(l,Y)
                dcdl_list.append(dcdl)
                dcdy_list.append(dcdy)
                loss_list.append(self.loss(l, Y))
                l_grad = dcdl
                y_grad = dcdy
                self.counter += 1


        return loss_list, dcdl_list, dcdy_list, l, Y
    
    def make_plots(self):
        x = np.linspace(1,self.counter,self.counter)
        fig, axis = plt.subplots(1,2,figsize=(10,5))
        axis[0].plot(x,self.loss_list)
        axis[0].set_title('loss')
        axis[0].set_yscale('log')
        axis[1].plot(x,self.dcdl_list, label='dcdl')
        axis[1].set_title('normalized gradient')
        axis[1].plot(x,self.dcdy_list, label='dcdy')
        axis[1].set_yscale('log')
        plt.legend()
        plt.show()
