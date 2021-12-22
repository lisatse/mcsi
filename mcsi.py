import random
import numpy as np
from biclustering import Construct_Binary_Clustering
import matplotlib.pyplot as plt
import scipy 
from functools import reduce
import os
import argparse
import faulthandler; faulthandler.enable()

class Winnow:
    def __init__(self,dims,i,string,data_normal=False,lr=None,data_source="data_", sideinfo=None,R_bound=True,use_graph=True): 
        self.dims = dims
        self.actual_dims = (self.dims[0]+self.dims[1],
            self.dims[0]+self.dims[1])
        self.sideinfo = sideinfo
        self.latent_dims=(float(string[0]),float(string[1]))
        self.R_bound = R_bound

        basepath = os.path.dirname(__file__) 
        if data_normal==True:
            data_fol = os.path.join(basepath,"data_"+string,str(dims[0]))
        else:
            data_fol = os.path.join(basepath,data_source,string,str(dims[0]))
        self.mat = np.load(os.path.join(data_fol,"bi_clust"+str(i)+".npy")) 
        
        self.psi_rows = self._construct_psi(os.path.join(data_fol,"laplacian_row"+str(i)+".npy"))
        self.psi_cols = self._construct_psi(os.path.join(data_fol,"laplacian_col"+str(i)+".npy"),row_flag=False)

        k = self.latent_dims[0]
        if use_graph is True:
            row_list = np.load(os.path.join(data_fol,"row_list1.npy"))
            col_list = np.load(os.path.join(data_fol,"col_list1.npy"))
            cut_row = row_list[i]
            cut_col = col_list[i]
        else:
            cut_row = k-1
            cut_col = k-1
        if self.sideinfo == "identity":
            D = self.dims[0]+self.dims[1]
        else:
            D= 4.*(cut_row*self.RL_row+k) + 4*(cut_col*self.RL_col +k) + 4.*k
        if lr is None:
            c=1.
            self.lr = np.sqrt(D*np.log(self.actual_dims[0])/ (2*c*self.dims[0]*self.dims[1]) ) 
        else:
            self.lr = lr
    
        diagW=np.ones(self.actual_dims[0]) *D/self.actual_dims[0] 
        self.W = np.diag(diagW)
        self.logW = np.diag(np.log(diagW))
        self.gamma = 1./np.sqrt(k)
        self.M = 0
        self.ilist = np.arange(self.dims[0]*self.dims[1])
        np.random.shuffle(self.ilist)
        self.ind=0
    
    def _construct_psi(self,laplacian_file,row_flag=True):
        laplacian = np.load(laplacian_file)
        if self.sideinfo == "identity":
            psi = np.eye(len(laplacian))/ np.sqrt(2)
            RL=1.
        else:
            Lplus = np.linalg.pinv(laplacian) 
            RL = np.amax(np.diag(Lplus))
            psi = Lplus + RL * np.ones_like(Lplus) 
            RM = 2*RL
            if self.sideinfo == "combined":
                psi = self._sqrt(psi + RM * np.eye(len(laplacian))) / (2*np.sqrt(RM))
            else:
                psi = self._sqrt(psi) / (np.sqrt(2*RM))
        if self.R_bound is True:
            self.RL = 4.
        else:
            if row_flag is True:
                self.RL_row = RL
            else:
                self.RL_col = RL
        return np.transpose(psi)

    def _exponentiate(self,X):
        eigvals,eigvecs  = np.linalg.eigh(X)
        exp_eig = np.exp(eigvals)
        return reduce(np.matmul,[eigvecs, np.diag(exp_eig),np.transpose(eigvecs)])
    
    def _sqrt(self,X):
        eigvals,eigvecs  = np.linalg.eigh(X)
        sqrt_eig = np.sqrt(eigvals)
        return reduce(np.matmul,[eigvecs, np.diag(sqrt_eig),np.transpose(eigvecs)])

    def _expected_mistakes(self,y_pred):
        sample = random.uniform(-self.gamma,self.gamma)
        ypred_binary = np.sign(y_pred-sample)
        return ypred_binary

    def _construct_X(self,i,j):
        x_i =self.psi_rows[i]
        x_j =self.psi_cols[j]
        x = np.concatenate( (x_i,x_j)) 
        self.X = np.outer(x,x)


    def _one_step(self):
        index=self.ilist[self.ind]
        i = int(np.floor(index/self.dims[0])) 
        j = int(index%self.dims[0])
        self._construct_X(i,j)
        y_act = self.mat[i,j]
        y_pred = np.trace(np.matmul(self.W,self.X)) -1.
        ypred_binary = self._expected_mistakes(y_pred) 
        if ypred_binary*y_act<=0:
            self.M += 1
        if y_pred*y_act< self.gamma:
            exp_term = self.lr * y_act  * self.X
            self.logW = self.logW + exp_term 
            self.W = self._exponentiate(self.logW)
        self.ind+=1
            

    def play_game(self,steps):
        self.mistakes_list = np.zeros(int(steps/10))
        i=0
        for step in range(steps):
            self._one_step()
            if (int(step+1))%10==0:
                self.mistakes_list[i] = self.M
                i+=1

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--string", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    dims_list = [400] 
    args = parser_args()
    string = args.string 
    basepath = os.path.dirname(__file__) 
    for dims in dims_list:
        folder = os.path.join(basepath,"results_diff_lr","results_"+string,str(dims))
        for i in range(3):
            winnow = Winnow((dims,dims),i,string,sideinfo=None,R_bound=False,use_graph=True)
            steps = dims*dims 
            winnow.play_game(steps)
            #plt.plot(range(int(steps/10)),winnow.mistakes_list,"+") 
            try:
                np.savetxt(os.path.join(folder,"scalar"+str(i)+".csv"),[winnow.M])
            except OSError:
                os.makedirs(folder)
                np.savetxt(os.path.join(folder,"scalar"+str(i)+".csv"),[winnow.M])
