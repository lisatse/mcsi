import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import os

class Construct_Binary_Clustering:
    def __init__(self,dims,latent_dims): 
        self.dims = dims
        self.latent_dims = latent_dims  

    def constr_latent_mat(self):                                                
        self.latent_mat = np.random.randint(low=0,high=2,size=self.latent_dims)
        zero_ind = np.argwhere(np.logical_not(self.latent_mat))
        for ind in zero_ind:
            ind = tuple(ind)
            self.latent_mat[ind] = -1.


    def constr_latent_map(self,latent_dim,dim):
        success = True
        latent_map_rows = np.random.randint(low=0,high = latent_dim,
            size=dim, dtype = np.int)
        for latent_d in range(latent_dim):
            nodes = np.argwhere(latent_map_rows==latent_d)
            if not list(nodes[:,0]):
                success = False
        return latent_map_rows,success
                

    def constr_one_hot(self,dim,latent_dim,row_flag=True):
        success = False
        while success is False:
            latent_map_rows, success = self.constr_latent_map(latent_dim,dim)
        if row_flag is True:
            one_hot_rows = np.zeros((dim,latent_dim))
        else:
            one_hot_rows = np.zeros((latent_dim,dim))
        for n in range(dim):
            lat_n = latent_map_rows[n]
            if row_flag is True:
                one_hot_rows[n,lat_n] = 1.
            else:
                one_hot_rows[lat_n,n] = 1.

        return latent_map_rows,one_hot_rows

    def mult_matrix(self):
        self.constr_latent_mat()
        self.latent_rows,self.one_hot_rows = self.constr_one_hot(self.dims[0],
                     self.latent_dims[0],row_flag=True)
        self.matrix = np.matmul(self.one_hot_rows, self.latent_mat)
        self.latent_cols,self.one_hot_cols = self.constr_one_hot(self.dims[1],
                     self.latent_dims[1],row_flag=False)
        self.matrix = np.matmul(self.matrix, self.one_hot_cols)
    

    

    def create_noisy_Laplacians(self, row_flag=True,prob=0.95):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.dims[0]))
        if row_flag==True:
            latent_dims = self.latent_dims[0]
            latent_list = self.latent_rows
        else:
            latent_dims = self.latent_dims[1]
            latent_list = self.latent_cols

        
        for n in range(self.dims[0]):
            for m in range(n+1,self.dims[0]): 
                draw = random.random()
                if latent_list[m] == latent_list[n]:
                    if draw <= prob:
                        self.G.add_edge(n,m)
                else:
                    if draw > prob:
                        self.G.add_edge(n,m)
        g = nx.connected_components(self.G)
        for i,gen in enumerate(g):
            if i ==0:
                gen1 = tuple(gen)
                node_center = random.choice(gen1) 
            if i> 0:
                node_side = random.choice(tuple(gen))
                self.G.add_edge(node_center,node_side)
        
#        nx.draw(self.G)
#        plt.savefig("graph.png",pos=nx.spring_layout(self.G))
        laplacian = nx.laplacian_matrix(self.G)
        laplacian = laplacian.toarray()
        self.G.clear()
        return laplacian

    def flip_labels(self,prob):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                draw = random.random()
                if draw > prob :
                    self.matrix[i,j] *= -1.

    def cut_size(self,laplacian, row=False):
        if row is True:
            out =  np.matmul(self.one_hot_rows.transpose(),laplacian)
            out = np.matmul(out,self.one_hot_rows)
        else:
            out =  np.matmul(self.one_hot_cols,laplacian)
            out = np.matmul(out,self.one_hot_cols.transpose())
        return np.trace(out)

def main():
    visualise = False 
    #dims_list = [200] 
    dims_list =list(range(20,220,20))+[250,300,400] 
    if visualise:
        dims_list = [400] 
    flip_labels  = True 
    noisy_laplacian = True 
    latent_dims = (9,9)
    #noiselist=[1.0]
    flip_noise=0.9
    noiselist  = [0.75, 1.0,0.5,0.875,0.9375,0.96875]
    
    for noise in noiselist:
        if flip_labels is True and noisy_laplacian is False:
            string = "_flip" +str(flip_noise)
        elif noisy_laplacian is True and flip_labels is False:
            string = "_noisy"+str(noise)
        elif noisy_laplacian is True and flip_labels is True:
            string = "flip"+str(flip_noise)+"_noisy"+str(noise)
        else:
            string= ""
        lat_str = str(latent_dims[0]) + str(latent_dims[1])
        
        for dims in dims_list:
            if visualise==False:
                folder = os.path.join("data_",lat_str+string,str(dims))
                steps=10
            if visualise==True:
                folder = os.path.join("data_visualise","data_"+lat_str+string,str(dims))
                steps=1

            row_list = np.zeros(steps)
            col_list = np.zeros(steps)

            for i in range(10):
                biclustering = Construct_Binary_Clustering((dims,dims),latent_dims)
                biclustering.mult_matrix()     
                if flip_labels is True:
                    biclustering.flip_labels(flip_noise)
                try:
                    np.save(os.path.join(folder,"bi_clust"+str(10+i)),biclustering.matrix)
                except OSError:
                    os.makedirs(folder)
                    np.save(os.path.join(folder,"bi_clust"+str(10+i)),biclustering.matrix)
                if noisy_laplacian is False:
                    laplacian_row = biclustering.create_noisy_Laplacians(row_flag=True,prob=1.0)
                    laplacian_col = biclustering.create_noisy_Laplacians(row_flag=False,prob=1.0)
                else: 
                    laplacian_row = biclustering.create_noisy_Laplacians(row_flag=True,prob = noise)
                    laplacian_col = biclustering.create_noisy_Laplacians(row_flag=False,prob = noise)
                
                    row_list[i] = biclustering.cut_size(laplacian_row,row=True)
                    col_list[i] = biclustering.cut_size(laplacian_col,row=False)
                if visualise:
                    np.save(os.path.join(folder,"latent_list"+str(10+i)), biclustering.latent_cols)
                np.save(os.path.join(folder,"laplacian_row"+str(10+i)), laplacian_row)
                np.save(os.path.join(folder,"laplacian_col"+str(10+i)), laplacian_col)
            row_list1=np.load(os.path.join(folder,"row_list.npy"))
            col_list1=np.load(os.path.join(folder,"col_list.npy"))
            row_list = np.concatenate((row_list1,row_list))
            col_list = np.concatenate((col_list1,col_list))
            np.save(os.path.join(folder,"row_list1"),row_list)
            np.save(os.path.join(folder,"col_list1"),col_list)
                
if __name__ == "__main__":
    main()
