
import torch
"""
We prioritze replays in the replay memory by TD Error (r + yV(S') - V(s))

"""

class PrioritizedMemory():
    def __init__(self, batch_size):
        
        self.priorities = torch.zeros((1,batch_size))
        self.D = []
        self.count = 0
        self.beta = 0.6
        self.batch_size = batch_size
        self.buffer_len = batch_size * 10

    def add(self, memory, priority):

        if self.count < self.buffer_len:

            self.priorities[self.count] = priority
            self.D.append(memory)
            self.count += 1
        else:
            
            remove_idx = self.priorities.argmin().item()
            self.priorities[remove_idx] = priority
            self.D[remove_idx] = memory

    def get(self):
        if self.count <= self.batch_size:
            return False
        probabilities = self.priorities / self.priorities.sum().item()

        samples = torch.multinomial(probabilities[:self.count], num_samples=self.batch_size)
        randoms = [self.D[sample] for sample in samples]
        priorities = [self.priorities[sample] for sample in samples]
        return randoms, priorities
    
    
    def update(self, idx, value):
        self.priorities[idx] = value
    
    def weights(self, indices):
        
        N = len(self.D)
        priorities= [self.priorities[i] for i in indices]

        weights = [((1/N) * (1/priority)) ** self.beta for priority in priorities]
        return weights
    

    @property
    def next_idx(self):
        if self.count < self.transitions:
            return self.count
        else:
            return self.priorities.argmin().item()
