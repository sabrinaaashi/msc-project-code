import sys
sys.path.append("/home/tereza/control01/tereza")

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp

import torch

def vecnorm(x, ord=2):
    '''Returns the norm of x'''
    if ord == float('inf'):
        return torch.max(torch.abs(x))
    elif ord == float('-inf'):
        return torch.min(torch.abs(x))
    else:
        return torch.sum(torch.abs(x)**ord, dim=0)**(1.0 / ord)

def bisection(f, uk , fprime, fk , gk , pk  ,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,traj , realizations,noise  ,max_iter=10000 , max_alpha = 5000 , c1=1e-10 ):

    #1. initialize
    gval = [None]
    gval_alpha = [None]
    minimal_alphastep=1e-17
    
    #2. test to see whether direction pk from Polak Ribiere method is actually a descet direction. If not use -gk, the negative gradient of the cost functional, instead.
    fkp_test = f(uk + 1e-13 * pk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,traj,realizations,noise)
    if fkp_test>fk or torch.isnan(fkp_test):
     #   print('Replaced PR descent direction with negative gradient at x0[10]',uk[10])
        pk = -gk

    #3. function calculating the cost functional for the control uk + alpha2*pk.
    def phi(alpha2):
        return f(uk + alpha2*pk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,traj,realizations,noise)
    
    def loop_for_stepsize(pk,alpha_ini):
    #4. initialize
        warning=3
        alpha1=alpha_ini #start with a large stepsize
        iteration=0
    
        #5. loop for finding the stepsize
        while warning==3:
            alpha1=alpha1/2
           # print(alpha1)
            if iteration > max_iter or alpha1 < minimal_alphastep:
                print('Bisection: minimal alphastep or max iteration reached:',alpha1)
                ukp=uk.clone()
                break
    
            if torch.max(torch.abs(uk + alpha1 * pk))<50: #set this condition, since for larger control values the network dynamics might diverge
                fkp = phi(alpha1)
               # print(alpha1,fk,fkp)
                if (fkp <= fk ): #
                    warning=0    
                #    print('found!')
                    ukp = uk + alpha1*pk 
                    break

            iteration += 1
        return warning , ukp , fkp , alpha1 

    alpha_ini=100
    warning , ukp , fkp , alpha1  = loop_for_stepsize(pk,alpha_ini)
    print(pk)
    if not isinstance(ukp, torch.Tensor):
        ukp = torch.tensor(ukp, dtype=torch.float32, requires_grad=True)
    elif not ukp.requires_grad:
        ukp.requires_grad_(True)


    #6.   
    if warning==3:
    #no stepsize found  . functional might be too flat
    #retry with different stepsizes
        alpha_ini=15
        warning , ukp , fkp , alpha1  = loop_for_stepsize(pk,alpha_ini)  

    if warning==3:
        gval_alpha[0]=None
        ukp=uk.clone()
        fkp=fk
        gkp=gk.clone()

    else:
    #stepsize found.
        gval_alpha[0]=alpha1
        ukp.retain_grad()
        gkp=fprime(ukp, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,traj,realizations,noise   ) 
    return ukp , fkp , gkp , gval_alpha[0] 

def direction(k , gk, gkp , pk):

    if k%100==0:
        betak=0
    else:
        gkdiff=gkp-gk
        #Polak-Ribiere method:
        betak = torch.max(torch.tensor(0), torch.dot(gkp, (gkdiff)) / torch.dot(gk, gk) )
        #print(betak)
        #Hestenes-Stiefel method:
      #  betak= max(0, np.dot(gkp, gkdiff) / np.dot(pk, gkdiff) )
    pkp = -gkp + betak*pk
    return pkp

def FR_algorithm(f,x0,fprime,traj,max_k=250,gtol=1e-4,**args):

    #1. load args
    tsteps = args['tsteps'] #timesteps, int
    dt = args['dt'] #stepsize of timesteps, float
    N = args['N'] #number of nodes, int
    d = args['d'] #dimension of oscillator dynamnode_ics, int
    alpha = args['alpha'] #parameter of FHN oscillator, float
    beta = args['beta'] #parameter of FHN oscillator, float
    gamma = args['gamma'] #parameter of FHN oscillator, float
    delta = args['delta'] #parameter of FHN oscillator, float
    epsilon = args['epsilon'] #parameter of FHN oscillator, float
    tau = args['tau'] #parameter of FHN oscillator, float
    mu = args['mu'] #parameter of FHN oscillator, float
    sigma = args['sigma'] #parameter of FHN oscillator, float
    A = args['A'] #adjacency matrix, array shape(N,N)
    I_p = args['I_p'] #weight of the precision term of the cost functional, float
    I_e = args['I_e'] #weight of the energy term of the cost functional, float
    I_s = args['I_s'] #weight of the sparsity term of the cost functional, float
    target_output = args['target_output'] #desired/target state, array shape(tsteps,N)
    node_ic= args['node_ic'] #initial conditions of the network dynamnode_ics, array shape(d,N),
    realizations=args['realizations']
    noise=args['noise']
    gk_list = []
    fk_list = []


    #2. Iitialization
    uk=x0
    fk=f(uk,tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,traj,realizations,noise)
    gk=fprime(uk, tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A,target_output,traj ,realizations,noise)
    gk_list.append(gk.clone().detach())
    fk_list.append(fk.clone().detach())
    pk=-gk.clone()
    warnflag=0
    k=1
    gnorm=vecnorm(gk,float('inf'))


    not_ready=True

    #3f. terminate if the solution is already optimal.
    if gnorm<=gtol*(1+fk):
        not_ready=False

    #3. Loop to find optimal solution. Wranflag=0: optimal solution is found, break. Warnflag=1: solution is not optimal yet. Warnflag=3: Negative derivative was o descent direction, Error, break. 
    while not_ready:
        #if k%1==0:
        #    print('iteration',k)
        #3a. When maximal iteration step number max_k is reached break, in order to save values and restart.
        if k>=max_k:
            warnflag=1
            print('Maximal iteration of FR algorithm reached.')
            break


        #3b. use bisection to calculate the stepsize alphak and the resulting new control ukp, the new cost functional fkp, its gradient gkp.
        ukp , fkp , gkp , alphak  = bisection(f , uk , fprime , fk , gk , pk , tsteps , d , dt , N , I_p , I_e , I_s , alpha , beta, gamma, delta, epsilon, tau, mu ,sigma, A, target_output,traj,realizations,noise)


        #3c. if alphak=None no stepsize was found for which f(uk)>f(uk+alphak*gk) holds. Something went wrong, the negative derivative was no descent direction. Error.
        if alphak==None:
            warnflag=3
            print('Problem with the linesearch.')
            break

        #3d. calculate the new descent direction.
        pkp = direction(k , gk, gkp , pk)

        #needed for 3f.
        gnorm=vecnorm(gkp,float('inf'))
        uknorm=vecnorm(uk-ukp,float('inf'))
        #print(gnorm,fkp)

        #3e. redifine
        uk=ukp.clone()
        fk=fkp#.clone()
        gk=gkp.clone()
        gk_list.append(gk.detach())
        fk_list.append(fk.detach())
        pk=pkp.clone()


        k+=1

        #3f. terminate the algorithm if the gradient becomes zero or the control does not change any more (because of box constrains on control).
        if gnorm<=gtol*(1+fkp) or uknorm<=1e-20:
            print("gnorm",gnorm,"uknorm",uknorm)
            break


    #4. return 
    return uk,fk,warnflag,gnorm,k,fk_list,gk_list

