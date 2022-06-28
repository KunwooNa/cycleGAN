import random
from attr import s 
import torch 


class ImageBuffer() :
    
    def __init__(self, buffer_size) :
        self.buffer_size = buffer_size 
        self.memory = []
        self.memory_counter = 0 

    def query(self, images) :
        returns = []
        for image in images : 
            image = torch.unsqueeze(image.data, 0)
            if self.memory_counter < self.buffer_size : 
                self.memory.append(image)
                returns.append(image)
                self.memory_counter += 1
            else : 
                if random.uniform(0, 1) < 0.5 : 
                    idx = random.randint(0, self.buffer_size - 1) 
                    previous_image = self.memory[idx].clone()
                    self.memory.append(previous_image)
                    self.memory[idx] = image
                else : 
                    returns.append(image)
        return returns 