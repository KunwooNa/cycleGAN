import random
# from attr import s 
import torch 


class ImageBuffer()  :
    
    def __init__(self, max_size)  :

        self.max_size = max_size 
        self.memory = [] 
        self.memory_cntr = 0 


    def query(self, images) :
        return_images = [] 
        for image in images  :
            image = torch.unsqueeze(image.data, 0) 

            if self.memory_cntr < self.max_size : 
                self.memory.append(image) 
                self.memory_cntr += 1
                return_images.append(image)

            else :
                p = random.uniform(0, 1) 
                if p > 0.5 : 
                    random_idx = random.randint(0, self.max_size - 1) 
                    tmp = self.memory[random_idx].clone()
                    self.memory[random_idx] = image 
                    return_images.append(tmp)
                
                else : 
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images 