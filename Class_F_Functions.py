import numpy as np
import sys
from scipy.spatial import distance



def scale_columns(points):
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    range = maxs - mins
    #print(range)
    return  1-(maxs- points)/range


'''
Method to take two equally-sized lists and return just the elements which lie 
on the Pareto frontier, sorted into order.
Default behaviour is to find the maximum for both X and Y, but the option is
available to specify maxX = False or maxY = False to find the minimum for either
or both of the parameters.
'''
def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y
                p_front.append(pair)      #  and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y
                p_front.append(pair)      #  and add them to the Pareto frontier

# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def igd(obtained, ideals):

    if np.isnan(np.min(obtained)):
        return 10.0**6
    igd_val = 0
    maxdist = 10.0**10
    for d in ideals:
        min_dist = maxdist
        for o in obtained:
            min_dist = min(min_dist, distance.euclidean(o, d))
        igd_val += min_dist
    return igd_val/len(ideals)




class F_Functions:
    Obj_Number  =      {"F1": 2, \
                        "F2": 2, \
                        "F3": 2, \
                        "F4": 2, \
                        "F5": 2, \
                        "F6": 3, \
                        "F7": 2, \
                        "F8": 2, \
                        "F9": 2 }

    Dec_Var_ranges =   {"F1": np.array([[0,1],[0,1],[0,1]]), \
                        "F2": np.array([[0,1],[-1,1],[-1,1]]), \
                        "F3": np.array([[0,1],[-1,1],[-1,1]]), \
                        "F4": np.array([[0,1],[-1,1],[-1,1]]), \
                        "F5": np.array([[0,1],[-1,1],[-1,1]]), \
                        "F6": np.array([[0,1],[0,1],[-2,2]]), \
                        "F7": np.array([[0,1],[0,1],[0,1]]), \
                        "F8": np.array([[0,1],[0,1],[0,1]]), \
                        "F9": np.array([[0,1],[-1,1],[-1,1]]) }

     
    def __init__(self,number_variables,function_name):
        self.n = number_variables
        self.fname = function_name
        self.n_obj =     self.Obj_Number[self.fname]
        self.DV_range =  self.Dec_Var_ranges[self.fname]
        self.MOP_Function = self.MOP_Functions_Dir[self.fname]
        self.PS_Function = self.PS_Functions_Dir[self.fname]




    def F1_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n)) 
        xvals[:,0] = x1
        for j in range(2,self.n+1):
            eval = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:,j-1] = x1**eval
        return xvals


    def F1_MOP_Evals(self,x):   
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
        val = np.zeros((N))
        for j in J1:
            eval = 0.5*(1.0+(3*(j-2)/(self.n-2)))      
            val = val +  (x[:,j-1]- x[:,0]**eval)**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val
        val = np.zeros((N))
        for j in J2:
            eval = 0.5*(1.0+(3*(j-2)/(self.n-2)))      
            val = val +  (x[:,j-1]- x[:,0]**eval)**2        
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*val
        return f1,f2


    def F2_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n)) 
        xvals[:,0] = x1
        for j in range(2,self.n+1):
            xvals[:,j-1] = np.sin(6*np.pi*x1+j*np.pi/self.n)        
        return xvals


    def F2_MOP_Evals(self,x): 
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)

        val = np.zeros((N))
        for j in J1:
            val = val +  (x[:,j-1]-np.sin(6*np.pi*x[:,0]+j*np.pi/self.n))**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val

        val = np.zeros((N))
        for j in J2:
            val = val +  (x[:,j-1]-np.sin(6*np.pi*x[:,0]+j*np.pi/self.n))**2
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*val
        return f1,f2


    def F3_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n)) 
        xvals[:,0] = x1   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
        for j in J1:
            xvals[:,j-1] = 0.8*x1*np.cos(6*np.pi*x1+j*np.pi/self.n) 

        for j in J2:
            xvals[:,j-1] = 0.8*x1*np.sin(6*np.pi*x1+j*np.pi/self.n) 
        return xvals


    def F3_MOP_Evals(self,x): 
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)

        val = np.zeros((N))
        for j in J1:
            val = val +  (x[:,j-1]- 0.8*x[:,0]*np.cos(6*np.pi*x[:,0]+j*np.pi/self.n) )**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val

        val = np.zeros((N))
        for j in J2:
            val = val +  (x[:,j-1]- 0.8*x[:,0]*np.sin(6*np.pi*x[:,0]+j*np.pi/self.n) )**2  
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*val
        return f1,f2


    def F4_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n)) 
        xvals[:,0] = x1   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
        for j in J1:
            xvals[:,j-1] = 0.8*x1*np.cos((6*np.pi*x1+j*np.pi/self.n)/3) 

        for j in J2:
            xvals[:,j-1] = 0.8*x1*np.sin(6*np.pi*x1+j*np.pi/self.n) 
        return xvals

    def F4_MOP_Evals(self,x): 
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
        val = np.zeros((N))
        for j in J1:
            val = val +  (x[:,j-1]- 0.8*x[:,0]*np.cos( (6*np.pi*x[:,0]+j*np.pi/self.n)/3) )**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val
        val = np.zeros((N))
        for j in J2:
            val = val +  (x[:,j-1]- 0.8*x[:,0]*np.sin(6*np.pi*x[:,0]+j*np.pi/self.n) )**2  
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*val
        return f1,f2

    def F5_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n)) 
        xvals[:,0] = x1   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
    
        for j in J1:
            xvals[:,j-1] = (0.3*(x1**2)*np.cos(24*np.pi*x1+4*j*np.pi/self.n)+0.6*x1)*np.cos(6*np.pi*x1+j*np.pi/self.n)

        for j in J2:
            xvals[:,j-1] = (0.3*(x1**2)*np.cos(24*np.pi*x1+4*j*np.pi/self.n)+0.6*x1)*np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals


    def F5_MOP_Evals(self,x): 
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)

        val = np.zeros((N))
        for j in J1:
            val = val + (x[:,j-1] - (0.3*(x[:,0]**2)*np.cos(24*np.pi*x[:,0]+4*j*np.pi/self.n)+0.6*x[:,0])*np.cos(6*np.pi*x[:,0]+
                                                                                                        j*np.pi/self.n) )**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val

        val = np.zeros((N))
        for j in J2:
            val = val +  (x[:,j-1]- (0.3*(x[:,0]**2)*np.cos(24*np.pi*x[:,0]+4*j*np.pi/self.n)+0.6*x[:,0])*np.sin(6*np.pi*x[:,0]
                                                                                                         +j*np.pi/self.n) )**2
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*val
        return f1,f2


    def F6_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n))
        xvals[:,[0,1]] = x1
        for j in range(3, self.n+1):
            sn = np.sin(2*np.pi * xvals[:, 0] + j*np.pi/self.n)
            xvals[:, j-1] = 2 * xvals[:, 1] * sn
        return xvals

    def F6_MOP_Evals(self,x):
        N = x.shape[0]

        J1 = np.arange(4, self.n+1, 3)
        J2 = np.arange(5, self.n+1, 3)
        J3 = np.arange(3, self.n+1, 3)

        val = np.zeros((N))
        for j in J1:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2

        f1 = np.cos(0.5*x[:, 0]*np.pi)*np.cos(0.5*x[:, 1]*np.pi)+2/J1.shape[0]*val

        val = np.zeros((N))
        for j in J2:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f2 = np.cos(0.5*x[:, 0]*np.pi)*np.sin(0.5*x[:, 1]*np.pi)+2/J2.shape[0]*val

        val = np.zeros((N))
        for j in J3:
            val = val + (x[:, j-1] - 2*x[:, 1] * np.sin(2*np.pi*x[:, 0] + j*np.pi/self.n))**2
        f3 = np.sin(0.5*x[:, 0]*np.pi) + 2/J3.shape[0]*val

        return f1, f2, f3


    def F7_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n))
        xvals[:,0] = x1
        for j in range(2,self.n+1):
            eval = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:,j-1] = x1**eval
        return xvals


    def F7_MOP_Evals(self,x): 
        N = x.shape[0]
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)

        val = np.zeros((N))
        for j in J1:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            val = val + 4*y**2 - np.cos(8*y*np.pi) + 1
        f1 = x[:, 0] + 2/J1.shape[0]*val

        val = np.zeros((N))
        for j in J2:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            val = val + 4*y**2 - np.cos(8*y*np.pi) + 1
        f2 = 1 - np.sqrt(x[:, 0]) + 2/J2.shape[0]*val

        return f1,f2

    def F8_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n))
        xvals[:,0] = x1
        for j in range(2,self.n+1):
            eval = 0.5*(1.0+(3*(j-2)/(self.n-2)))
            xvals[:,j-1] = x1**eval
        return xvals

    def F8_MOP_Evals(self,x): 
        N = x.shape[0]
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)
        sum = np.zeros((N))
        prod = np.ones((N))
        for j in J1:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            sum = sum + y**2
            prod = prod * np.cos(20*y*np.pi/np.sqrt(j)) + 2
        f1 = x[:,0] + (2.0/J1.shape[0])*(4*sum-2*prod)
        val = np.zeros((N))
        prod = np.ones((N))
        for j in J2:
            y = x[:, j-1]-x[:, 0]**(0.5*(1.0+(3*(j-2)/(self.n-2))))
            sum = sum + y**2
            prod = prod * np.cos(20*y*np.pi/np.sqrt(j)) + 2
        f2 = 1 - np.sqrt(x[:,0]) + (2.0/J2.shape[0])*(4*sum-2*prod)
        return f1,f2

    def F9_MOP(self,x1):
        xvals = np.zeros((x1.shape[0],self.n))
        xvals[:,0] = x1
        for j in range(2,self.n+1):
            xvals[:,j-1] = np.sin(6*np.pi*x1+j*np.pi/self.n)
        return xvals


    def F9_MOP_Evals(self,x): 
        N = x.shape[0]   
        J1 = np.arange(3,self.n+1,2)
        J2 = np.arange(2,self.n+1,2)

        val = np.zeros((N))
        for j in J1:
            val = val +  (x[:,j-1]-np.sin(6*np.pi*x[:,0]+j*np.pi/self.n))**2
        f1 = x[:,0] + (2.0/J1.shape[0])*val

        val = np.zeros((N))
        for j in J2:
            val = val +  (x[:,j-1]-np.sin(6*np.pi*x[:,0]+j*np.pi/self.n))**2
        f2 = 1 - np.power(x[:,0], 2) + (2.0/J2.shape[0])*val
        return f1,f2




    PS_Functions_Dir = {"F1": F1_MOP, \
                        "F2": F2_MOP, \
                        "F3": F3_MOP, \
                        "F4": F4_MOP, \
                        "F5": F5_MOP, \
                        "F6": F6_MOP, \
                        "F7": F7_MOP, \
                        "F8": F8_MOP, \
                        "F9": F9_MOP }
    
    MOP_Functions_Dir = {"F1": F1_MOP_Evals, \
                         "F2": F2_MOP_Evals, \
                         "F3": F3_MOP_Evals, \
                         "F4": F4_MOP_Evals, \
                         "F5": F5_MOP_Evals, \
                         "F6": F6_MOP_Evals, \
                         "F7": F7_MOP_Evals, \
                         "F8": F8_MOP_Evals, \
                         "F9": F9_MOP_Evals  }
    
    
    def Evaluate_MOP_Function(self,x):
        if self.fname == "F6":
            f1,f2,f3 =  self.MOP_Function(self,x)
            return f1,f2,f3
        else:
            f1,f2 =  self.MOP_Function(self,x)
            return f1,f2


    def Generate_MOP_samples(self,k):   
        all_x = np.zeros((k,self.n))
        for i in range(self.n):
            if i<2:
                all_x[:,i] = np.random.uniform(low=self.DV_range[i,0], high=self.DV_range[i,1], size=k)
            else:
                all_x[:,i] = np.random.uniform(low=self.DV_range[2,0], high=self.DV_range[2,1], size=k)        
        return all_x

    
    def Generate_PS_samples(self,k): 
        if  self.n_obj==2:
            all_ps = np.random.uniform(low=self.DV_range[0,0], high=self.DV_range[0,1], size=k)   
        elif  self.n_obj==3:
            all_ps = np.zeros((k,2))
            all_ps[:,0] = np.random.uniform(low=self.DV_range[0,0], high=self.DV_range[0,1], size=k)   
            all_ps[:,1] = np.random.uniform(low=self.DV_range[1,0], high=self.DV_range[1,1], size=k)   
        PS_samples =  self.PS_Function(self,all_ps)
        return PS_samples




