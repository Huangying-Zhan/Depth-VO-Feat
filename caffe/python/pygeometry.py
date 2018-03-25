import caffe
import numpy as np
import time


class SE3_Generator_KITTI(caffe.Layer):
    """
    SE3_Generator takes 6 transformation parameters (se3) and generate corresponding 4x4 transformation matrix
    Input: 
        bottom[0] | se3 | shape is (batchsize, 6, 1, 1)
    Output: 
        top[0]    | SE3 | shape is (batchsize, 1, 4, 4)
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one input to compute transformation matrix.")

        # Define variables
        self.batchsize = bottom[0].num
        self.threshold = 1e-12

    def reshape(self, bottom, top):
        # check input dimension
        if bottom[0].count%6 != 0: #bottom.shape = (batchsize,6)
            raise Exception("Inputs must have the correct dimension.")
        # Output is 4x4 transformation matrix
        top[0].reshape(bottom[0].num,1,4,4)

    def forward(self, bottom, top):
        # Define skew matrix of so3, .size = (batchsize,1,3,3)
        self.uw = bottom[0].data[:,:3]

        self.uw_x = np.zeros((self.batchsize,1,3,3))
        self.uw_x[:,0,0,1] = -self.uw[:,2,0,0]
        self.uw_x[:,0,0,2] = self.uw[:,1,0,0]
        self.uw_x[:,0,1,0] = self.uw[:,2,0,0]
        self.uw_x[:,0,1,2] = -self.uw[:,0,0,0]
        self.uw_x[:,0,2,0] = -self.uw[:,1,0,0]
        self.uw_x[:,0,2,1] = self.uw[:,0,0,0]

        # Get translation lie algebra
        self.ut = bottom[0].data[:,3:]
        self.ut = np.reshape(self.ut, (self.batchsize,1,3,1))
        # Calculate SO3 and T, i.e. rotation matrix (batchsize,1,3,3) and translation matrix (batchsize,1,1,3)
        self.R = np.zeros((self.batchsize,1,3,3))
        self.R[:,0] = np.eye(3)
        self.theta = np.linalg.norm(self.uw,axis=1) #theta.size = (batchsize,1)
        for i in range(self.batchsize):
            if self.theta[i]**2 < self.threshold:
                self.R[i,0] += self.uw_x[i,0]
                # self.V[i,0] += 0.5 * self.uw_x[i,0]
                continue
            else:
                c1 = np.sin(self.theta[i])/self.theta[i]
                c2 = 2*np.sin(self.theta[i]/2)**2/self.theta[i]**2
                c3 = ((self.theta[i] - np.sin(self.theta[i]))/self.theta[i]**3)**2
                self.R[i,0] += c1*self.uw_x[i,0] + c2*np.dot(self.uw_x[i,0],self.uw_x[i,0])
                # self.V[i,0] += c2*self.uw_x[i,0] + c3*np.dot(self.uw_x[i,0],self.uw_x[i,0])
        
        # Calculate output
        top[0].data[:,:,:3,:3] = self.R
        # top[0].data[:,:,:3,3] = np.matmul(self.V, self.ut)[:,:,:,0]
        # Rt implementation
        top[0].data[:,:,:3,3] = np.matmul(self.R, self.ut)[:,:,:,0]
        top[0].data[:,:,3,3] = 1


    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            # top[0].diff .shape is (batchsize,1,4,4)
            dLdT = top[0].diff[:,:,:3,3].copy() #batchsize,1,3
            dLdT = dLdT[:,np.newaxis]

            # Rt implementation for DLdut is dLdT x R
            # dLdut = np.matmul(dLdT, self.V)
            dLdut = np.matmul(dLdT, self.R)
            bottom[0].diff[:,3:,0,0] = dLdut[:,0,0]
            # Gradient correction for dLdR. '.' R also affect T, need update dLdR
            grad_corr = np.matmul(np.swapaxes(dLdT, 2, 3), np.swapaxes(self.ut, 2, 3))  # from (b,hw,4,1) to (b,4,hw,1)


            # dLduw
            dLdR = top[0].diff[:,:,:3,:3]
            dLdR += grad_corr
            dLduw = np.zeros((self.batchsize,3))
            # for theta less than threshold
            generators = np.zeros((3,3,3))
            generators[0] = np.array([[0,0,0],[0,0,1],[0,-1,0]])
            generators[1] = np.array([[0,0,-1],[0,0,0],[1,0,0]])
            generators[2] = np.array([[0,1,0],[-1,0,0],[0,0,0]])
            for index in range(3):
                I3 = np.zeros((self.batchsize,1,3,3))
                I3[:,0] = np.eye(3)
                ei = np.zeros((self.batchsize,1,3,1))
                ei[:,0,index] = 1
                cross_term = np.matmul(self.uw_x, np.matmul(I3-self.R,ei))
                cross = np.zeros((self.batchsize,1,3,3))
                cross[:,0,0,1] = -cross_term[:,0,2,0]
                cross[:,0,0,2] = cross_term[:,0,1,0]
                cross[:,0,1,0] = cross_term[:,0,2,0]
                cross[:,0,1,2] = -cross_term[:,0,0,0]
                cross[:,0,2,0] = -cross_term[:,0,1,0]
                cross[:,0,2,1] = cross_term[:,0,0,0]
                self.dRduw_i = np.zeros((self.batchsize,1,3,3))
                for j in range(self.batchsize):
                    if self.theta[j]**2 < self.threshold:
                        self.dRduw_i[j] = generators[index]
                    else:
                        self.dRduw_i[j,0] = np.matmul((self.uw[j,index]*self.uw_x[j,0] + cross[j,0])/(self.theta[j]**2), self.R[j,0])
                dLduw[:,index]=np.sum(np.sum(dLdR*self.dRduw_i,axis=2),axis=2)[:,0]

            bottom[0].diff[:,:3,0,0] = dLduw

