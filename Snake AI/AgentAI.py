# -*- coding: utf-8 -*-
"""
Description: AI Agent that plays the game of snake

Sugestion: 
    Run this program in a conda enviroment or any other virtual environment 
    with pygame and pytorch installed ('conda activate pygame_env')
"""

import numpy as np
import random
import torch
from SnakeGame import AISnakeGame, Direction, Point
from collections import deque
from DeepQNetModel import DeepQNet, Trainer
import matplotlib.pyplot as plt
from IPython import display

MAX_MEM= 100000 #Set a maximum capacity of memory
BATCH_SIZE = 1000 #number of samples for our memory
ALPHA = 0.001 #Learning rate

class Agent:
    
    def __init__(self):
        self.ep = 0  # coef of randomness
        self.gamma = 0.9 # discount rate
        self.num_games = 0
        #deque will automatically do popleft() if it exceeds the MAX_MEM
        self.memory = deque(maxlen=MAX_MEM) 
        """To do: model trainer """
        self.model = DeepQNet(11,256,3)
        self.trainer = Trainer(self.model, gamma=self.gamma,alpha=ALPHA)
    
    def getAction(self, state):

        #Make random moves
        self.ep= 80 - self.num_games
        action = [0,0,0]
        if random.randint(0,200) < self.ep:
            #srl = straight (0), right(1), left(2)
            srl = random.randint(0,2)
            action[srl] = 1
        else:
            st = torch.tensor(state, dtype=torch.float)
            #get a prediction
            prediction = self.model(st)
            srl = torch.argmax(prediction).item()
            action[srl] = 1
        
        return action
    
    def getState(self, game):
        #get coordinates of head of the snake and the posible next points
        head = game.snake[0]
        head_x = head.x
        head_y = head.y
        p_up = Point(head.x, head.y - 20)
        p_right = Point(head.x + 20, head.y)
        p_down = Point(head.x, head.y + 20)
        p_left = Point(head.x - 20, head.y)
        #get coordinates of food
        food_x = game.food.x
        food_y = game.food.y
        #get the current direction the snake is pointing at
        dir_up = game.direction == Direction.UP
        dir_right = game.direction == Direction.RIGHT
        dir_down = game.direction == Direction.DOWN
        dir_left = game.direction == Direction.LEFT
       
        #Check if there is danger around the snake and where
        #Use the is_collision function from SnakeGame.py
        if dir_up:
            danger_straight = game.is_collision(p_up)
            danger_right = game.is_collision(p_right)
            danger_left = game.is_collision(p_left)
        elif dir_right:
            danger_straight = game.is_collision(p_right)
            danger_right = game.is_collision(p_down)
            danger_left = game.is_collision(p_up)
        elif dir_down:
            danger_straight = game.is_collision(p_down)
            danger_right = game.is_collision(p_left)
            danger_left = game.is_collision(p_right)
        elif dir_left:
            danger_straight = game.is_collision(p_left)
            danger_right = game.is_collision(p_up)
            danger_left = game.is_collision(p_down)
       
        #Check where the food is respect to the head of the snake
        food_right = head_x < food_x
        food_left = head_x > food_x
        food_up = head_y > food_y
        food_down = head_y < food_y
        
        #Make the state list
        state = [danger_straight, danger_right, danger_left,
                 dir_left, dir_right, dir_up, dir_down,
                 food_left, food_right, food_up, food_down]
        return np.array(state, dtype=int)
    
    
    def remember(self, state, next_state, action, reward, lose):

        self.memory.append((state, next_state, action, reward, lose))
    
    def shortMem(self,state, next_state, action, reward, lose):

        self.trainer.train_step(state, next_state, action, reward, lose)
    
    def longMem(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #List of tuples
        else:
            mini_sample = self.memory
       
        #get every sate together, every next_sate together, etc. 
        states, next_states, actions, rewards, loses = zip(*mini_sample)
        self.trainer.train_step(states, next_states, actions, rewards, loses) 
    

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def train():
    scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = AISnakeGame()
    
    #Game training loop
    while True:
        # get old state
        state = agent.getState(game)
        #get action
        action = agent.getAction(state)
        #Model predict
        reward, game_over, score = game.play_step(action)
        new_state = agent.getState(game)
        
        #Train the short term memory
        agent.shortMem(state, new_state, action, reward, game_over)
        #call remember function
        agent.remember(state, new_state, action, reward, game_over)
        
        if game_over:
            game.reset()
            agent.num_games += 1
            #Train long term memory 
            agent.longMem()
            
            if score > best_score:
                best_score = score
                agent.model.save()
            
            print('Game:', agent.num_games, 'Score:', score, 'Best Score:', best_score)
            
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)
            

if __name__ == '__main__':
    train()