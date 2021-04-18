#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#
#import torch
#import torch.optim as optim
#import torch.nn as nn
import numpy as np


#*****************************************************************************#
#                                                                             #
#   description:                                                              #
#   class that implements contract logic for Federated Distillation.          #
#                                                                             #
#*****************************************************************************#
class Server():
    def __init__(self, n_samples, n_classes, n_workers, alpha=1.0):
        # meta-information about dataset
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_workers = n_workers
        self.alpha = alpha
        # Collects worker predictions Cij matrix
        self.wr_predict = []
        self.label_dist = []
        # Stores results of aggregated predictions
        self.majorityVote = []
        self.rewardShares = []


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used to compute the aggregated labels and reward.       #
    #                                                                     #
    #---------------------------------------------------------------------#
    def aggregate_and_compute_reward(self):
        # local variables for computation of reward & majority
        rewardShare = np.zeros(self.n_workers)
        Votes = np.zeros((self.n_samples, self.n_classes), dtype=int)
        Sum = np.zeros(self.n_classes, dtype=float)
        
        # aggregate and store predictions
        for (prediction, freq) in zip(self.wr_predict, self.label_dist):
            Sum += freq
            for j, sample_predict in enumerate(prediction):
                Votes[j, sample_predict] += 1
            
        # compute reward for each worker
        for j in range(0, self.n_samples):
            for i in range(0, self.n_workers):
                # Compute R_i
                Ri = (1.0/(self.n_workers*self.n_samples)) * (Sum - self.label_dist[i])
                t0 = 0
                # Reward worker i for each peer p
                for p in range(0, self.n_workers):
                    # Skip if same worker
                    if i == p:
                        continue
                    # Compute reward
                    t0 += ((1.0/Ri[self.wr_predict[i][j]]) - 1) if self.wr_predict[i][j] == self.wr_predict[p][j] else -1
                # Reward Share for worker i
                rewardShare[i] += self.alpha * (1.0/(self.n_workers-1)) * t0
        
        # Compute the majority vote
        self.majorityVote.append(np.argmax(Votes, axis=-1).astype("uint8"))
        self.rewardShares.append(rewardShare)

        # Clear local variables and reset caches for next round
        del Votes, Sum
        self.clear_caches()
        

    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used to clear local memory.                             #
    #                                                                     #
    #---------------------------------------------------------------------#            
    def clear_caches(self):
        # delete old data
        del self.wr_predict, self.label_dist
        # allocate new buffers
        self.wr_predict = []
        self.label_dist = []


    #---------------------------------------------------------------------#
    #                                                                     #
    #   Functions used by workers to communicate with server.             #
    #                                                                     #
    #---------------------------------------------------------------------#        
    def receive_prediction(self, w_id, predictions, freq):
        self.wr_predict.append(predictions)
        self.label_dist.append(freq)


    def global_labels(self):
        return self.majorityVote[-1] if len(self.majorityVote) > 0 else None
    
        
