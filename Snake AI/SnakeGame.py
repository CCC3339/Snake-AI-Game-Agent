# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 19:50:50 2021

Description: Snake pygame based on the code made by Python Engineer

Link to his code: 
    https://github.com/python-engineer/python-fun/tree/master/snake-pygame
    
@author: Daniel Dorantes
"""

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

#initialize the pygame
pygame.init()
font = pygame.font.SysFont('arial', 20)

"""
Checkist to what to change of the game
    -Reset game so that the AI can keep playing
    -Implement the reward system
    -Change play function so it gets an action play(action) returns direction
    -Keep track of the current game_iteration
    -Change in the is_collision function
"""

#Define class of direction
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 255, 100)
GREEN2 = (0, 255, 100)
BLACK = (0,0,0)
BLOCK_SIZE = 20
SPEED = 40

class AISnakeGame:
    
    #Intialize things need for the game
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        # initial state of the snake will be UP
        self.direction = Direction.UP
        
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.setFood()
        self.frame_iteration = 0
        
    def setFood(self):
        x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.setFood()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
        # 2. move
        self.move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward=0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward= -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            reward= 10
            self.score += 1
            self.setFood()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def move(self, action):
        #Action: [straigt, right, left]
        def_direction = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        index_dir = def_direction.index(self.direction)
        
        #Move Straight
        if np.array_equal(action, [1,0,0]): 
            direction = def_direction[index_dir]
        elif np.array_equal(action, [0,1,0]):
            new_index = (index_dir+1)%4
            direction= def_direction[new_index]
        else: #Action is equal to [0,0,1] Left
            new_index = (index_dir-1)%4
            direction= def_direction[new_index]
        
        self.direction = direction
        
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            