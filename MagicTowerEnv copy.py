from typing import List
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from gymnasium.envs.registration import register
from setuptools import setup
class MagicTowerEnv(gym.Env):
     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
     
     def __init__(self,render_mode = "human",width = 7,height = 8):
      super().__init__()
      self.width = width
      self.height = height
      # wall:-1
      # coin/key:4/5
      # enemy:2
      # exit:3
      # agent:1
      self.origin_map = np.transpose(np.array([
         [ 4, 0,-1, 3, 0, 2, 0],
         [ 0, 2,-1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0,-1,-1],
         [-1,-1,-1,-1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 2],
         [ 0,-1,-1, 0,-1, 4, 0],
         [ 2, 0,-1, 0,-1,-1, 0],
         [ 0, 4,-1, 1, 0, 0, 0],
         ]))

      

      self.start_pos = [3,7]
      self.agent_pos = self.start_pos.copy()

      self.max_HP = 4
      self.curr_HP = self.max_HP

      self.max_coin_num = 3
      self.curr_coin_num = self.max_coin_num

      self.max_enemy_num = 4
      self.curr_nemy_num = self.max_enemy_num

      #set key pos
      self.if_have_key =False
      self.key_pos_seed= {
            0: (0,0),
            1: (5,5),
            2: (1,7),
          }
      x,y = self.key_pos_seed[random.randint(0,2)]
      self.curr_map = self.origin_map.copy()
      self.curr_map[x,y] = 5


      #self.observation_space = spaces.Dict({
      #      "map": spaces.Box(-1,5,(width, height),np.int32),#map
      #      "if_have_key": spaces.Discrete(2),  #if_have_key
      #      "hp": spaces.Discrete(5)  #hp
      #    })

      self.observation_space = spaces.Box(low=-1, high=5,
                                            shape=(3, width, height))

          
          # "right", "up", "left", "down"
      self.action_space = spaces.Discrete(4)
      

      assert render_mode is None or render_mode in self.metadata["render_modes"]
      self.render_mode = render_mode
      self.window = None
      self.clock = None

     def _get_obs(self):#未订正，应改为3chanle网络
          #return{"map":self.curr_map, "if_have_key":self.if_have_key, "hp":self.curr_HP}
        shape = (self.width,self.height)
        if self.if_have_key:
            key_array = np.ones(shape)
        else:
            key_array = np.zeros(shape)
        
        hp_array = np.full(shape, self.curr_HP)
        assert self.curr_map.shape ==  key_array.shape == hp_array.shape 
        obs_array = np.stack((self.curr_map, key_array, hp_array ), axis=0)
        return obs_array





     def _get_info(self):
          return{"pos":self.agent_pos,
                 "if_have_key":self.if_have_key,
                 "enemy num": self.curr_nemy_num,
                 "hp":self.curr_HP,
                 "coin num":self.curr_coin_num }

     def reset(self,seed=None, options=None):
           self.if_have_key =False
           self.curr_HP = self.max_HP
           x,y = self.key_pos_seed[random.randint(0,2)]
           self.curr_map = self.origin_map.copy()
           self.curr_map[x,y] = 5
           self.agent_pos = self.start_pos.copy()
           self.curr_coin_num = self.max_coin_num
           self.curr_nemy_num = self.max_enemy_num

           return self._get_obs() , self._get_info()
      
     
     def step(self, action):
          next_x= self.agent_pos[0]
          next_y =self.agent_pos[1]
          reward = 0
          terminated = False

          if(action == 0):#up
              next_y-=1
          elif(action == 1):#down
              next_y+=1
          elif(action == 2):#right
              next_x+=1
          elif(action == 3):#left
              next_x-=1
          
          #print(str(self.agent_pos )+ "->" +str(action) + "->"  +str([next_x,next_y] ))
          if(next_x < 0 or next_x >=7 or next_y < 0 or next_y >=8):
              reward -=1
          else:
              # wall:-1
              if(self.curr_map[next_x,next_y] == -1):
                  reward -=1
              # way
              if(self.curr_map[next_x,next_y] == 0):
                  reward -=0.1  
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
              # coin
              if(self.curr_map[next_x,next_y] == 4):
                  reward +=20
                  
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
              #key
              if(self.curr_map[next_x,next_y] == 5):
                  reward +=50
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
                  self.if_have_key = True
              #exit
              if(self.curr_map[next_x,next_y] == 3):
                  if(self.if_have_key):
                    reward +=500
                    terminated = True
                    self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                    self.curr_map[next_x,next_y] = 1
                    self.agent_pos = [next_x,next_y]
                  else:
                    reward -=10
              #enemy
              if(self.curr_map[next_x,next_y] == 2):
                  win_rate = random.randint(0,100)
                  if(win_rate >= 50):#win
                     reward +=20
                     self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                     self.curr_map[next_x,next_y] = 1
                     self.agent_pos = [next_x,next_y]
                  else:
                     self.curr_HP -= 1
                     if(self.curr_HP <= 0):
                         reward -=100
                         terminated = True
                     else:
                         reward -=0.5
                         self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                         self.curr_map[next_x,next_y] = 1
                         self.agent_pos = [next_x,next_y]
          
          observation = self._get_obs()
          info = self._get_info()

          
          
          
          return observation, reward, terminated, False, info
     

     def render(self):
         if self.window is None:
             pygame.init()
             pygame.display.init()
             self.window = pygame.display.set_mode(
                (70, 80)
            )
         if self.clock is None :
            self.clock = pygame.time.Clock()
         canvas = pygame.Surface((70, 80))
         canvas.fill((255, 255, 255))
         pix_square_size = 10
         for x in range(0,6):
             for y in range(0,7):
                 color =(255,255,255)
                 if(self.curr_map[x][y] == -1):
                     color = (0,0,0)
                 if(self.curr_map[x][y] == 1):
                     color = (0,0,255)
                 if(self.curr_map[x][y] == 3):
                     color = (0,255,0)
                 if(self.curr_map[x][y] == 2):
                     color = (255,0,0)
                 if(self.curr_map[x][y] == 4):
                     color = (255,255,0)  
                 if(self.curr_map[x][y] == 5):
                     color = (255,160,0) 
                 pygame.draw.rect(canvas,color,
                                  pygame.rect(
                                      pix_square_size*(self.agent_pos[0],self.agent_pos[1]),
                                     (pix_square_size,pix_square_size)
                                     ))
         self.window.blit(canvas, canvas.get_rect())
         pygame.event.pump()
         pygame.display.update()

         self.clock.tick(self.metadata["render_fps"])
     
            
register(
    id='MagicTowerEnv-v0',
    entry_point='MagicTowerEnv:MagicTowerEnv',
    max_episode_steps=1000,
)
           
          
          




