import os
import numpy as np

def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)


def construct_psi(laplacian_file):
    laplacian = np.load(laplacian_file)
    Lplus = np.linalg.pinv(laplacian) 
    RL = np.amax(np.diag(Lplus))
    psi_plus = Lplus + RL * np.ones_like(Lplus) 
    RM = np.amax(np.diag(psi_plus))
    return RM

def check_Lplus(laplacian_file):
    L = np.load(laplacian_file)
    Lplus = np.linalg.pinv(L)
    mat = np.matmul(L,Lplus)
    mat = np.matmul(mat,L)
    assert(np.allclose(L, mat, rtol=1e-05, atol=1e-08)==True)

def main():
    string_list = ["99_noisy0.5","99_noisy0.75","99_noisy0.875","99_noisy0.9375","99_noisy1.0"]

    for string in string_list:
        print(string)
        dims=400
        basepath = os.path.dirname(__file__) 
        folder = os.path.join(basepath,"data_"+string,str(dims))
        for i in range(3):
            check_Lplus(os.path.join(folder,"laplacian_row" + str(i) + ".npy"))
            check_Lplus(os.path.join(folder,"laplacian_col" + str(i) + ".npy"))
            R_row =construct_psi(os.path.join(folder,"laplacian_row" + str(i) + ".npy"))
            R_col =construct_psi(os.path.join(folder,"laplacian_col" + str(i) + ".npy"))
            print("R row is ",R_row,"R col is ",R_col)

if __name__ == "__main__":
    main()
