# -*- coding: utf-8 -*-
"""
Description: Code that implements the Deep Q Learning model using Pytorch as
            the framework.

Sugestion: 
    Run this program in a conda enviroment or any other virtual environment 
    with pygame and pytorch installed ('conda activate pygame_env')
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DeepQNet(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):

        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        #Create Layer 1 of the neural network
        self.lin1 = nn.Linear(input_dims, hidden_dims)
        #Create Layer 2 of the neural network
        self.lin2 = nn.Linear(hidden_dims, output_dims)
    
    def forward(self, state):

        x = F.relu(self.lin1(state))
        actions = self.lin2(x)
        return actions
    
    def save(self, file_name = 'model.pth'):

        #Create a path for a model folder
        path = './model'
        #Check if the directory folder already exists
        if not os.path.exists(path):
            #Create the model folder 
            os.makedirs(path)
        file = os.path.join(path, file_name)
        #Save the file of the model
        T.save(self.state_dict(), file) 

class Trainer:
    """ Class for making the optimization of the model"""
    def __init__(self, model, gamma, alpha):

        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        #Define the loss function
        self.loss = nn.MSELoss() 
        #Create an optimizer 
        self.opt  = optim.Adam(model.parameters(), lr=self.alpha)
        
    def train_step(self,state, next_state, action, reward, lose):

        #Create every argument except done, into a pytorch tensor so it can
        #be used in the model (n,x)
        state = T.tensor(state,dtype=T.float)
        next_state = T.tensor(next_state,dtype=T.float)
        action = T.tensor(action,dtype=T.long)
        reward = T.tensor(reward,dtype=T.float)
        #Handle multiple sizes or dimensions for the long term memory function
        #First check if state has only one dimension
        if len(state.shape)==1:
            #Re-shape the arguments so we have it like (1,x)
            #(n,x) where n is the number of batches and x is the value

            state = T.unsqueeze(state,0)
            next_state = T.unsqueeze(next_state,0)
            action= T.unsqueeze(action,0)
            reward = T.unsqueeze(reward,0)
            #make lose argument a tuple with one value
            lose = (lose, )
            
        #Calculate Bellman equation 
        #obtain the prediction (Q) of current state and the of the next state
        predQ = self.model(state)
        next_predQ = self.model(next_state)
        #Create a clone of the prediction Q called target
        target = predQ.clone()
        for i in range (len(lose)):
            new_Q = reward[i]
            if not lose[i]:
                #Apply the Bellman equation if not lose
                #NewQ = r + gamma * max(next predicted Q value)
                new_Q=reward[i]+self.gamma*T.max(next_predQ[i])
            
            target[i][T.argmax(action).item()] = new_Q
        
        self.opt.zero_grad()
        loss = self.loss(target,predQ)
        #Create a backward propagation and update the gradients
        loss.backward()
        
        self.opt.step()
        