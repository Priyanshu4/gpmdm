# Copyright 2022 by Fabio Amadio.
# All rights reserved.
# This file is part of the cgpdm_lib,
# and is released under the "GNU General Public License".
# Please see the LICENSE file included in the package.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import pickle
from termcolor import colored, cprint
from pathlib import Path


class CGPDM(torch.nn.Module):
    """
    Class Aware Gaussian Process Dynamical Model

    Attributes
    ----------
    dtype : torch.dtype
        data type of torch tensors

    device : torch.device
        device on which a tensors will be allocated

    D : int
        observation space dimension

    d : int
        desired latent space dimension
    
    n_classes : int 
        number of classes

    dyn_target : string
        dynamic map function target ('full' or 'delta')

    dyn_step_back : int
        memory of dynamic map function (1 or 2)

    y_log_lengthscales : torch.nn.Parameter
        log(lengthscales) of Y GP kernel

    y_log_lambdas : torch.nn.Parameter
        log(signal inverse std) of Y GP kernel

    y_log_sigma_n : torch.nn.Parameter
        log(noise std) of Y GP kernel

    x_log_lengthscales : torch.nn.Parameter
        log(lengthscales) of X GP kernel

    x_log_lambdas : torch.nn.Parameter
        log(signal inverse std) of X GP kernel

    x_log_sigma_n : torch.nn.Parameter
        log(noise std) of X GP kernel

    x_log_lin_coeff : torch.nn.Parameter
        log(linear coefficients) of X GP kernel

    X : torch.nn.Parameter
        latent states

    sigma_n_num_X : double 
        additional noise std for numerical issues in X GP

    sigma_n_num_Y : double
        additional noise std for numerical issues in Y GP

    observations_list : list(double)
        list of observation sequences

    Kx_inv : torch.Tensor
        inverted dynamics map kernel matrix

    Ky_inv : torch.Tensor
        inverted latent map kernel matrix

    """
    def __init__(self, D, d, n_classes, dyn_target, dyn_back_step,
                 y_lambdas_init, y_lengthscales_init, y_sigma_n_init,
                 x_lambdas_init, x_lengthscales_init, x_sigma_n_init,
                 x_lin_coeff_init, flg_train_y_lambdas = True,
                 flg_train_y_lengthscales = True, flg_train_y_sigma_n = True,
                 flg_train_x_lambdas = True, flg_train_x_lengthscales = True,
                 flg_train_x_sigma_n = True, flg_train_x_lin_coeff = True,
                 sigma_n_num_Y = 0., sigma_n_num_X = 0.,
                 dtype = torch.float64, device = torch.device('cpu')):
        """
        Parameters
        ----------
        D : int
            observation space dimension

        d : int
            latent space dimension

        n_classes : int 
            number of classes

        dyn_target : string
            dynamic map function target ('full' or 'delta')
        
        dyn_back_step : int
            memory of dynamic map function (1 or 2)

        y_lambdas_init : double
            initial signal std for GP Y (dimension: D)

        y_lengthscales_init : double
            initial lengthscales for GP Y (dimension: d)

        y_sigma_n_init : double
            initial noise std for GP Y (dimension: 1)
        
        x_lambdas_init : double
            initial signal std for GP X (dimension: d)

        x_lengthscales_init : double
            initial lengthscales for GP X (dimension: d*dyn_back_step)

        x_sigma_n_init : double
            initial noise std for GP X (dimension: 1)

        x_lin_coeff_init : double
            initial linear coefficients for GP X (dimension: d*dyn_back_step+1)

        flg_train_y_lambdas : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_y_lengthscales : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_y_sigma_n : boolean (optional)
            requires_grad flag for associated parameter
                 
        flg_train_x_lambdas : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_lengthscales : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_sigma_n : boolean (optional)
            requires_grad flag for associated parameter

        flg_train_x_lin_coeff : boolean (optional)
            requires_grad flag for associated parameter

        sigma_n_num_Y : double (optional)
            additional noise std for numerical issues in X GP

        sigma_n_num_X : double (optional)
            additional noise std for numerical issues in X GP

        dtype: torch.dtype (optional)
            data type of torch tensors

        device: torch.device (optional)
            device on which a tensors will be allocated

        """
        super(CGPDM,self).__init__()

        # torch parameters
        self.dtype = dtype
        self.device = device

        # observation dimension
        self.D = D
        # desired latent dimension
        self.d = d
        # number of classes
        self.n_classes = n_classes
        # dynamic model target
        self.dyn_target = dyn_target
        # dynamic model input
        self.dyn_back_step = dyn_back_step

        # Set Y-kernel parameters
        self.y_log_lengthscales = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(y_lengthscales_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_y_lengthscales)
        self.y_log_lambdas = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(y_lambdas_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_y_lambdas)
        self.y_log_sigma_n = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(y_sigma_n_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_y_sigma_n)

        # Set X-kernel parameters
        self.x_log_lengthscales = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(x_lengthscales_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_x_lengthscales)
        self.x_log_lambdas = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(x_lambdas_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_x_lambdas)
        self.x_log_sigma_n = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(x_sigma_n_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_x_sigma_n)
        self.x_log_lin_coeff = torch.nn.Parameter(torch.tensor(
            np.log(np.abs(x_lin_coeff_init)),
            dtype = self.dtype,
            device = self.device),
            requires_grad = flg_train_x_lin_coeff)

        # additional noise variance for numerical issues
        self.sigma_n_num_Y = sigma_n_num_Y
        self.sigma_n_num_X = sigma_n_num_X

        # Initialize observations
        self.class_aware_observations_list = [[] for _ in range(self.n_classes)]

    def set_evaluation_mode(self):
        """
        Set the model in evaluation mode
        """
        self.flg_trainable_list = []
        for p in self.parameters():
            p.requires_grad = False

    def set_training_mode(self, model = 'all'):
        """
        Set the model in training mode
        
        Parameters
        ----------

        model : string ['all', 'latent' or 'dynamics'] (optional)
            'all' set all requires_grad to True
            'latent' set only GP Y parameters to True
            'dynamics' set only GP X parameters to True
        """
        if model == 'all':
            for i, p in enumerate(self.parameters()):
                p.requires_grad = True
        elif model == 'latent':
            self.y_log_lengthscales.requires_grad = True
            self.y_log_lambdas.requires_grad = True
            self.y_log_sigma_n.requires_grad = True
            self.x_log_lengthscales.requires_grad = False
            self.x_log_lambdas.requires_grad = False
            self.x_log_sigma_n.requires_grad = False
            self.x_log_lin_coeff.requires_grad = False
        elif model == 'dynamics':
            self.y_log_lengthscales.requires_grad = False
            self.y_log_lambdas.requires_grad = False
            self.y_log_sigma_n.requires_grad = False
            self.x_log_lengthscales.requires_grad = True
            self.x_log_lambdas.requires_grad = True
            self.x_log_sigma_n.requires_grad = True
            self.x_log_lin_coeff.requires_grad = True
        else:
            raise ValueError('model must be \'all\', \'latent\' or \'dynamics\'')      

    def add_data(self, Y, class_index):
        """
        Add observation data to self.observations_list

        Parameters
        ----------

        Y : double
            observation data (dimension: N x D)

        class_index : int
            class index of the observation data

        """
        if Y.shape[1] != self.D:
            raise ValueError('Y must be a N x D matrix collecting observation data!')

        self.class_aware_observations_list[class_index].append(Y)

    @property
    def observations_list(self):
        """
        Return the list of all observations

        Return
        ------
        observations_list : list of all observations
        """
        return [seq for class_seqs in self.class_aware_observations_list for seq in class_seqs]

    def get_M(self):
        """
        Construct the class-specific block diagonal matrix M.

        Returns
        -------
        M : torch.Tensor
            Block diagonal matrix M with each block corresponding to a class.
        """
        # Total number of data points
        num_data_points = sum([len(seq) for class_seqs in self.class_aware_observations_list for seq in class_seqs])
        
        # Initialize M as a zero matrix
        M = torch.zeros((num_data_points, num_data_points), dtype=self.dtype, device=self.device)
        
        # Offset to track the starting index for each class block
        offset = 0
        
        for class_seqs in self.class_aware_observations_list:
            # Flatten all sequences for the current class
            class_data_points = sum([len(seq) for seq in class_seqs])
            
            # Fill the diagonal block for the current class
            M[offset:offset + class_data_points, offset:offset + class_data_points] = torch.eye(class_data_points, dtype=self.dtype, device=self.device)
            
            # Update offset for the next block
            offset += class_data_points

        return M

    def get_M_for_class(self, class_index):
        """
        Construct the block diagonal matrix M_c for a specific class.

        Parameters
        ----------
        class_index : int
            Index of the class for which to construct M_c.

        Returns
        -------
        M_c : torch.Tensor
            Block diagonal matrix M_c for the specified class, with zeros elsewhere.
        """
        # Total number of data points
        num_data_points = sum([len(seq) for class_seqs in self.class_aware_observations_list for seq in class_seqs])
        
        # Initialize M_c as a zero matrix
        M_c = torch.zeros((num_data_points, num_data_points), dtype=self.dtype, device=self.device)
        
        # Offset to track the starting index for each class block
        offset = 0
        
        for i, class_seqs in enumerate(self.class_aware_observations_list):
            # Flatten all sequences for the current class
            class_data_points = sum([len(seq) for seq in class_seqs])
            
            if i == class_index:
                # Fill the diagonal block for the specified class
                M_c[offset:offset + class_data_points, offset:offset + class_data_points] = torch.eye(class_data_points, dtype=self.dtype, device=self.device)
                break  # Stop after filling the relevant block
            
            # Update offset for the next block
            offset += class_data_points

        return M_c


    def get_y_kernel(self, X1, X2, flg_noise = True):
        """
        Compute the latent mapping kernel (GP Y)
        
        Parameters
        ----------

        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_y(X1,X2)

        """
        return self.get_rbf_kernel(X1, X2, self.y_log_lengthscales, self.y_log_sigma_n, self.sigma_n_num_Y, flg_noise)

    def get_x_kernel(self, X1, X2, flg_noise = True):
        """
        Compute the latent dynamic kernel (GP X)

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_x(X1,X2) 

        """
        return self.get_rbf_kernel(X1, X2, self.x_log_lengthscales, self.x_log_sigma_n, self.sigma_n_num_X, flg_noise) + \
               self.get_lin_kernel(X1, X2, self.x_log_lin_coeff)   

    def get_rbf_kernel(self, X1, X2, log_lengthscales_par, log_sigma_n_par,
                       sigma_n_num = 0, flg_noise = True):
        """
        Compute RBF kernel on X1, X2 points (without signal variance)

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lengthscales_par : tensor(dtype)
            log(lengthscales) RBF kernel

        log_sigma_n_par : tensor(dtype)
            log(noise std)  RBF kernel

        sigma_n_num : double
            additional noise std for numerical issues

        flg_noise : boolean (optional)
            add noise to kernel matrix

        Return
        ------
        K_rbf(X1,X2)

        """

        if flg_noise:
            N = X1.shape[0]
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par)) + \
                torch.exp(log_sigma_n_par)**2 * torch.eye(N, dtype = self.dtype, device = self.device) + \
                sigma_n_num**2 * torch.eye(N, dtype = self.dtype, device = self.device)

        else:
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par))

    def get_weighted_distances(self, X1, X2, log_lengthscales_par):
        """
        Computes (X1-X2)^T*Sigma^-2*(X1-X2) where Sigma = diag(exp(log_lengthscales_par))

        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lengthscales_par : tensor(dtype)
            log(lengthscales)
        
        Return
        ------
        dist = (X1-X2)^T*Sigma^-2*(X1-X2)

        """
        lengthscales = torch.exp(log_lengthscales_par)

        X1_sliced = X1 / lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim = 1, keepdim = True)
        X2_sliced = X2 / lengthscales
        X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim = 1, keepdim = True)
        dist = X1_squared + X2_squared.transpose(dim0 = 0, dim1 = 1) - \
            2 * torch.matmul(X1_sliced,X2_sliced.transpose(dim0 = 0, dim1 = 1))

        return dist


    def get_lin_kernel(self, X1, X2, log_lin_coeff_par):
        """
        Computes linear kernel on X1, X2 points: [X1,1]^T*Sigma*[X2,1] where Sigma=diag(exp(log_lin_coeff_par)) 
        
        Parameters
        ----------
        
        X1 : tensor(dtype)
            1st GP input points

        X2 : tensor(dtype)
            2nd GP input points

        log_lin_coeff_par : tensor(dtype)
            log(linear coefficients)

        Return
        ------
        K_lin(X1,X2)

        """
        Sigma = torch.diag(torch.exp(log_lin_coeff_par)**2)
        X1 = torch.cat([X1,torch.ones(X1.shape[0], 1, dtype = self.dtype, device = self.device)], 1)
        X2 = torch.cat([X2,torch.ones(X2.shape[0], 1, dtype = self.dtype, device = self.device)], 1)
        return torch.matmul(X1, torch.matmul(Sigma, X2.transpose(0,1)))


    def get_y_neg_log_likelihood(self, Y, X, N, M):
        """
        Compute latent negative log-likelihood Ly
        
        Parameters
        ----------

        Y : tensor(dtype)
            observation matrix

        X : tensor(dtype)
            latent state matrix

        N : int
            number of data points

        M : torch.Tensor
            Block diagonal matrix M for class-specific decorrelation.
    
        Return
        ------
        L_y = D/2*log(|K_y(X,X)|) + 1/2*trace(K_y^-1*Y*W_y^2*Y) - N*log(|W_y|)

        """
        K_y = self.get_y_kernel(X,X)
        U, info = torch.linalg.cholesky_ex(K_y, upper = True)
        U_inv = torch.inverse(U)
        Ky_inv = torch.matmul(U_inv,U_inv.t())
        log_det_K_y = 2 * torch.sum(torch.log(torch.diag(U)))

        W = torch.diag(torch.exp(self.y_log_lambdas))
        W2 = torch.diag(torch.exp(self.y_log_lambdas)**2)
        log_det_W = 2 * torch.sum(self.y_log_lambdas)

        Y_W2_Y = torch.linalg.multi_dot([Y, W2, Y.transpose(0, 1)])

        return self.D / 2 * log_det_K_y + \
            1 / 2 * torch.trace(torch.mm(Ky_inv, M*Y_W2_Y)) \
            - N * log_det_W

    def get_x_neg_log_likelihood(self, Xout, Xin, N):
        """
        Compute dynamics map negative log-likelihood Lx
        
        Parameters
        ----------

        Xout : tensor(dtype)
            dynamics map output matrix

        Xin : tensor(dtype)
            dynamics map input matrix

        N : int
            number of data points
        
        Return
        ------
        L_x = d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*W_x^2*Xout) - (N-dyn_back_step)*log(|W_x|)

        """
        
        K_x = self.get_x_kernel(Xin, Xin)
        U, info = torch.linalg.cholesky_ex(K_x, upper = True)
        U_inv = torch.inverse(U)
        Kx_inv = torch.matmul(U_inv, U_inv.t())
        log_det_K_x = 2 * torch.sum(torch.log(torch.diag(U)))

        W = torch.diag(torch.exp(self.x_log_lambdas))
        W2 = torch.diag(torch.exp(self.x_log_lambdas)**2)
        log_det_W = 2 * torch.sum(self.x_log_lambdas)

        return self.d / 2 * log_det_K_x + 1 / 2 * \
            torch.trace(torch.linalg.multi_dot([Kx_inv, Xout, W2, Xout.transpose(0,1)])) \
            - Xin.shape[0] * log_det_W

    def get_Xin_Xout_matrices(self, X = None, target = None, back_step = None):
        """
        Compute GP input and output matrices (Xin, Xout) for GP X

        Parameters
        ----------

        X : tensor(dtype) (optional)
            latent state matrix

        target : string (optional)
            dynamic map function target ('full' or 'delta')
        
        back_step : int (optional)
            memory of dynamic map function (1 or 2)
        
        Return
        ------
        Xin : GP X input matrix

        Xout : GP X output matrix

        start_indeces : list of sequences' start indeces

        """
        if X == None:
            X = self.X
        if target == None:
            target = self.dyn_target
        if back_step == None:
            back_step = self.dyn_back_step

        X_list = []
        x_start_index = 0
        start_indeces = []
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X[x_start_index:x_start_index + sequence_length,:])
            start_indeces.append(x_start_index)
            x_start_index = x_start_index + sequence_length           

        if target == 'full' and back_step == 1:
            # in: x(t)
            Xin = X_list[0][0:-1,:]
            # out: x(t+1)
            Xout = X_list[0][1:,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][0:-1,:]), 0)
                Xout = torch.cat((Xout, X_list[j][1:,:]), 0)

        elif target == 'full' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:],X_list[0][0:-2,:]), 1)
            # out: x(t+1)
            Xout = X_list[0][2:,:]
            for j in range(1, len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:],X_list[j][0:-2,:]), 1)), 0)
                Xout = torch.cat((Xout, X_list[j][2:,:]),0)

        elif target == 'delta' and back_step == 1:
            # in: x(t)
            Xin = X_list[0][0:-1,:]
            # out: x(t+1)-x(t)
            Xout = X_list[0][1:,:] - X_list[0][0:-1,:]
            for j in range(1, len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][0:-1,:]), 0)
                Xout = torch.cat((Xout, X_list[j][1:,:] - X_list[j][0:-1,:]), 0)

        elif target == 'delta' and back_step == 2:
            # in: [x(t), x(t-1)]
            Xin = torch.cat((X_list[0][1:-1,:],X_list[0][0:-2,:]), 1)
            # out: x(t+1)-x(t)
            Xout = X_list[0][2:,:] - X_list[0][1:-1,:]
            for j in range(1,len(self.observations_list)):
                Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:],X_list[j][0:-2,:]), 1)), 0)
                Xout = torch.cat((Xout, X_list[j][2:,:] - X_list[j][1:-1,:]), 0)
        
        else:
            raise ValueError('target must be either \'full\' or \'delta\' \n back_step must be either 1 or 2')

        return Xin, Xout, start_indeces


    def gpdm_loss(self, Y, N, M, balance=1):
        """
        Calculate GPDM loss function L = L_y + beta * L_x.

        Parameters
        ----------
        Y : tensor(dtype)
            Observation matrix.

        X : tensor(dtype)
            Latent state matrix.

        N : int
            Number of data points.

        M : torch.Tensor
            Block diagonal matrix M for class-specific decorrelation.

        balance : float, optional
            Balance factor beta for L_x (default is 1).

        Returns
        -------
        loss : float
            Total GPDM loss.
        """
        # Get input-output matrices for dynamics
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        
        lossY = self.get_y_neg_log_likelihood(Y, self.X, N, M)
        lossX = self.get_x_neg_log_likelihood(Xout, Xin, N)

        loss = lossY + balance*lossX

        return loss

    def init_X(self):
        """
        Initalize latent variables matrix with PCA
        """
        Y = self.get_Y()
        pca = PCA(n_components = self.d)
        X0 = pca.fit_transform(Y)

        # set latent variables as parameters
        self.X = torch.nn.Parameter(torch.tensor(X0, dtype = self.dtype, 
                            device=  self.device), requires_grad=True)

        # init inverse kernel matrices
        Ky = self.get_y_kernel(self.X, self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper = True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv, U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin, Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper = True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv, U_inv.t())

    def get_Y(self):
        """
        Create observation matrix Y from observations_list

        Return
        ------

        Y : observation matrix
        """
        observation = np.concatenate(self.observations_list, 0)

        # self.meanY = np.mean(observation,0)
        self.meanY = 0
        Y = observation - self.meanY
        return Y

    def train_adam(self, num_opt_steps, num_print_steps = 0,
                   lr = 0.01, balance = 1):
        """
        Optimize model with Adam

        Parameters
        ----------

        num_opt_steps : int 
            number of optimization steps

        num_print_steps : int
            number of steps between printing info

        lr : double
            learning rate

        balance : double
            balance factor for gpdm_loss

        Return
        ------

        losses : list of loss evaluated

        """

        if num_print_steps != 0:
            print('\n### Model Training (Adam) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype = self.dtype, device = self.device)
        M = self.get_M()

        self.set_training_mode('all')

        # define optimizer
        f_optim = lambda p : torch.optim.Adam(p, lr = lr)
        optimizer = f_optim(self.parameters())

        t_start = time.time()
        losses = []
        for epoch in range(num_opt_steps):
            optimizer.zero_grad()
            loss = self.gpdm_loss(Y, N, M, balance)
            loss.backward()
            if torch.isnan(loss):
                cprint('Loss is nan', 'red')
                break

            optimizer.step()

            losses.append(loss.item())

            if (num_print_steps != 0) and epoch % num_print_steps == 0:
                print('\nGPDM Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(loss.item()))
                t_stop = time.time()
                print('Update time:',t_stop - t_start)
                t_start = t_stop

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X, self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv, U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin, Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv, U_inv.t())

        return losses

    def get_latent_sequences(self):
        """
        Return the latent trajectories associated to each observation sequence recorded

        Return
        ------

        X_list : list of latent states associated to each observation sequence
        """
        X_np = self.X.clone().detach().cpu().numpy()
        X_list = []
        x_start_index = 0
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X_np[x_start_index:x_start_index + sequence_length,:])
            x_start_index = x_start_index + sequence_length

        return X_list
    
    def get_X_for_class(self, class_index):
        """
        Return the entire X matrix for a specific class

        Parameters
        ----------

        class_index : int
            Index of the class for which to return the X matrix
        """
        data_points_per_class = [sum(len(seq) for seq in self.class_aware_observations_list[i]) for i in range(self.n_classes)]

        start_index = sum(data_points_per_class[:class_index])
        end_index = start_index + data_points_per_class[class_index]

        return self.X[start_index:end_index, :]

    def map_x_to_y(self, Xstar, flg_noise = False):
        """
        Map Xstar to observation space: return mean and variance
        
        Parameters
        ----------

        Xstar : tensor(dtype)
            input latent state matrix 

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------

        mean_Y_pred : mean of Y prediction

        diag_var_Y_pred : variance of Y prediction


        """

        Y_obs = self.get_Y()
        Y_obs = torch.tensor(Y_obs, dtype = self.dtype, device = self.device)

        Ky_star = self.get_y_kernel(self.X, Xstar,False)

        mean_Y_pred = torch.linalg.multi_dot([Y_obs.t(), self.Ky_inv,Ky_star]).t()
        diag_var_Y_pred_common = self.get_y_diag_kernel(Xstar, flg_noise) - \
            torch.sum(torch.matmul(Ky_star.t(), self.Ky_inv) * Ky_star.t(), dim = 1)
        y_log_lambdas = torch.exp(self.y_log_lambdas)**-2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1) * y_log_lambdas.unsqueeze(0)

        return mean_Y_pred + torch.tensor(self.meanY, dtype = self.dtype, device = self.device), diag_var_Y_pred

    def get_y_diag_kernel(self, X, flg_noise = False):
        """
        Compute only the diagonal of the latent mapping kernel GP Y

        Parameters
        ----------

        X : tensor(dtype)
            latent state matrix

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------
        GP Y diag covariance matrix

        """

        n = X.shape[0]
        if flg_noise:
            return torch.ones(n, dtype = self.dtype, device = self.device) + torch.exp(self.y_log_sigma_n)**2 + self.sigma_n_num_Y**2
        else:
            return torch.ones(n, dtype = self.dtype, device = self.device)

    def map_x_dynamics(self, Xstar, flg_noise = False):
        """
        Map Xstar to GP dynamics output

        Parameters
        ----------

        Xstar : tensor(dtype)
            input latent state matrix 

        flg_noise : boolean
            add noise to kernel matrix

        Return
        ------

        mean_Xout_pred : mean of Xout prediction

        diag_var_Xout_pred : variance of Xout prediction

        """
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        
        Kx_star = self.get_x_kernel(Xin, Xstar,False)
   
        mean_Xout_pred = torch.linalg.multi_dot([Xout.t(), self.Kx_inv, Kx_star]).t()
        diag_var_Xout_pred_common = self.get_x_diag_kernel(Xstar, flg_noise) - \
            torch.sum(torch.matmul(Kx_star.t(), self.Kx_inv) * Kx_star.t(), dim = 1)
        x_log_lambdas = torch.exp(self.x_log_lambdas)**-2
        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1) * x_log_lambdas.unsqueeze(0)

        return mean_Xout_pred, diag_var_Xout_pred

    def get_x_diag_kernel(self, X, flg_noise = False):
        """
        Compute only the diagonal of the dynamics mapping kernel GP Y

        Parameters
        ----------

        X : tensor(dtype)
            latent state matrix

        flg_noise : boolean
            add noise to prediction variance
        
        Return
        ------
        GP X diag covariance matrix

        """

        n = X.shape[0]
        Sigma = torch.diag(torch.exp(self.x_log_lin_coeff)**2)
        X = torch.cat([X,torch.ones(X.shape[0],1, dtype = self.dtype, device = self.device)],1)
        if flg_noise:
            return torch.ones(n, dtype = self.dtype, device = self.device) +\
                   torch.exp(self.x_log_sigma_n)**2 + self.sigma_n_num_X**2 +\
                   torch.sum(torch.matmul(X, Sigma)*(X), dim = 1)
        else:
            return torch.ones(n, dtype = self.dtype, device = self.device) + \
                   torch.sum(torch.matmul(X, Sigma)*(X), dim=1)

    def get_next_x(self, gp_mean_out, gp_out_var, Xold, flg_sample = False):
        """
        Predict GP X dynamics output to next latent state

        Parameters
        ----------

        gp_mean_out : tensor(dtype)
            mean of the GP X dynamics output

        gp_out_var : tensor(dtype)
            variance of the GP X dynamics output

        Xold : tensor(dtype)
            present latent state

        flg_noise : boolean
            add noise to prediction variance

        Return
        ------

        Predicted new latent state

        """

        distribution = Normal(gp_mean_out, torch.sqrt(gp_out_var))  
        if self.dyn_target == 'full':
            if flg_sample:
                return distribution.rsample()
            else:
                return gp_mean_out

        if self.dyn_target == 'delta':
            if flg_sample:
                return Xold + distribution.rsample()
            else:
                return Xold + gp_mean_out

    def save(self, config_dict_path, state_dict_path):
        """
        Save model into two pickle objects

        Parameters
        ----------

        config_dict_path : string
            save path for the config_dict

        state_dict_path : string
            save path for the state_dict

        """

        torch.save(self.state_dict(), state_dict_path)
        config_dict={}
        config_dict['class_aware_observations_list'] = self.class_aware_observations_list
        config_dict['dyn_target'] = self.dyn_target
        config_dict['dyn_back_step'] = self.dyn_back_step
        config_dict['D'] = self.D
        config_dict['d'] = self.d
        config_dict['sigma_n_num_X'] = self.sigma_n_num_X
        config_dict['sigma_n_num_Y'] = self.sigma_n_num_Y
        pickle.dump(config_dict, open(config_dict_path, 'wb'))
        cprint("\nGPDM config dict saved in "+config_dict_path, "green")
        cprint("GPDM state dict saved in "+state_dict_path, "green")


    def load(self, config_dict, state_dict, flg_print = False):
        """
        Load (previously initialized) model

        Parameters
        ----------

        config_dict : pathlike or dict
            configuration dictionary

        config_dict : pathlike or collections.OrderedDict
            model state dictionary

        flg_print : bool (optional)
            flag to print loaded state_dict (default is False)
        """

        if isinstance(config_dict, (str, Path)):
            config_dict = pickle.load(open(config_dict, 'rb'))

        if isinstance(state_dict, (str, Path)):
            state_dict = torch.load(state_dict)

        self.class_aware_observations_list = config_dict['class_aware_observations_list']
        self.init_X()
        self.load_state_dict(state_dict)

        # save inverse kernel matrices after training
        Ky = self.get_y_kernel(self.X,self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = torch.matmul(U_inv,U_inv.t())
        
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx = self.get_x_kernel(Xin,Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = torch.matmul(U_inv,U_inv.t())

        cprint("\nGPDM correctly loaded", "green")

        if flg_print:
            print("Loaded params:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor])
