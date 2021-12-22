import os
import numpy as np
import matplotlib.pyplot as plt

def biclustered_mat(folder):
    file_name = np.load(os.path.join("data_visualise",folder, "400", "bi_clust0.npy"))
    plt.imshow(file_name,cmap="viridis")
    plt.savefig(folder+"biclustering.pdf")


def rotate_laplacian(laplacian,latent_list):
    latent_dims = np.amax(latent_list)
    shuffled_laplacian = np.zeros_like(laplacian)
    shuffled_laplacian2 = np.zeros_like(laplacian)
    for dim in range(len(laplacian)):
        laplacian[dim,dim] = 0.
    i = 0
    for latent_dim in range(latent_dims+1):
        indices  = np.argwhere(latent_list==latent_dim)[:,0]
        for (num,ind) in enumerate(indices):
            shuffled_laplacian[i] = laplacian[ind]
            i +=1
    j=0
    for latent_dim in range(latent_dims+1):
        indices  = np.argwhere(latent_list==latent_dim)[:,0]
        for (num,ind) in enumerate(indices):
            shuffled_laplacian2[:,j] = shuffled_laplacian[:,ind]
            j +=1

    return shuffled_laplacian2


def side_info(folder):
    main_fol = os.path.join("data_visualise",folder,"400")
    laplacian= np.load(os.path.join(main_fol, "laplacian_col0.npy"))
    latent_list= np.load(os.path.join(main_fol, "latent_list0.npy"))
    shuffled_laplacian = rotate_laplacian(laplacian, latent_list)
    plt.imshow(shuffled_laplacian,cmap="viridis")
    plt.savefig(folder+"laplacian.pdf")



def main():
    biclustered_mat("data_99flip0.95_noisy1.0")
    side_info("data_99flip0.95_noisy0.75")
    side_info("data_99flip0.95_noisy0.875")
    side_info("data_99flip0.95_noisy0.9375")

main()
