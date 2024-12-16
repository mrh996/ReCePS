
import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cvxpy as cp
import warnings
from sklearn import preprocessing

warnings.filterwarnings("ignore")
def get_cvar_cert_time_t(estimate, t, eps,sigma):
	erf = scipy.special.erf(math.sqrt(t+1) * eps/(2*math.sqrt(2)*sigma))
	cvar = 1. if estimate > erf else estimate/erf
	return cvar * erf

def get_exact_time_t(estimate, t, eps,sigma):
	return norm.cdf(norm.ppf(estimate) - math.sqrt(t+1) * eps/(sigma))

def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
def get_exact_total(estimate, eps,sigma):
	return norm.cdf(norm.ppf(estimate) -  eps/(sigma))

def _hoeffding_lcb(mean: float,  N: int, alpha: float) -> float:
	return max(mean - math.sqrt(math.log(1./alpha)/(2*N)),0)
def get_hs_bound(sigma,norm_bound:float, beta:float):
        
        a = norm_bound/(2*sigma) - np.log(beta)*sigma/(2*norm_bound)
        b = -norm_bound/(2*sigma) - np.log(beta)*sigma/(2*norm_bound)
        HS_bound = norm.cdf(a)-beta*norm.cdf(b)-max(1-beta,0)
        return max(0,HS_bound)

def optimizer(epsilon_0: float, epsilon_1: float, beta_0 : float, beta_1 : float,\
        pa: float, pb: float, pc: float) -> float :
        """
            Lagrange Dual : 

                lambda_0*[pa*f_0(ra)+pb*f_0(rb)+pc*f_0(rc)-epsilon_0] + 
                lambda_1*[pa*f_1(ra)+pb*f_1(rb)+pc*f_1(rc)-epsilon_1] +
                k*(1-pa*ra-pb*rb-pc*rc) + pa*ra - pb*rb 

        """

        print("p : ",pa,pb,pc)
        ##### ra, rb, rc #########
        ra = cp.Variable()
        rb = cp.Variable()
        rc = cp.Variable()
        ##########################

        ####### f-function ######################################
        f_0_ra = cp.maximum(ra-beta_0,0) - cp.maximum(1-beta_0,0)
        f_0_rb = cp.maximum(rb-beta_0,0) - cp.maximum(1-beta_0,0)
        f_0_rc = cp.maximum(rc-beta_0,0) - cp.maximum(1-beta_0,0)
        
        f_1_ra = cp.maximum(ra-beta_1,0) - cp.maximum(1-beta_1,0)
        f_1_rb = cp.maximum(rb-beta_1,0) - cp.maximum(1-beta_1,0)
        f_1_rc = cp.maximum(rc-beta_1,0) - cp.maximum(1-beta_1,0)
        
        #########################################################

        ####### lambda_0, lambda_1, k ###############
        lambda_0 = 0.0
        lambda_1 = 0.0
        k = 0.0
        #############################################
        
        n_iter = 50
        step_size = 0.1
        end_point = 0.001

        for i in range(n_iter):

            Lagrange_aug = lambda_0*(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0) + \
                lambda_1*(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1) + \
                k*(1-pa*ra-pb*rb-pc*rc) + pa*ra - pb*rb + cp.square(1-pa*ra-pb*rb-pc*rc) + \
                cp.square(cp.maximum(0,pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1)) + \
                cp.square(cp.maximum(0,pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0))
            
            cp.Problem(cp.Minimize(Lagrange_aug)).solve(solver = cp.ECOS)
            
            o_lambda_0 = lambda_0
            lambda_0 += step_size*(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0).value
            lambda_0 = max(lambda_0,0)

            o_lambda_1 = lambda_1
            lambda_1 += step_size*(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1).value
            lambda_1 = max(lambda_1,0)

            o_k = k
            k += step_size*(1-pa*ra-pb*rb-pc*rc).value

            if abs(o_lambda_0-lambda_0)<end_point and abs(o_lambda_1-lambda_1)<end_point \
            and abs(o_k-k)<end_point :
                break
        
        """ debug
        print("lambda_0 = ",lambda_0)
        print("pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0 = ",(pa*f_0_ra+pb*f_0_rb+pc*f_0_rc-epsilon_0).value)
        print("lambda_1 = ",lambda_1)
        print("pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1 = ",(pa*f_1_ra+pb*f_1_rb+pc*f_1_rc-epsilon_1).value)
        print("k = ",k)
        print("1-pa*ra-pb*rb-pc*rc",(1-pa*ra-pb*rb-pc*rc).value)
        """
        return (pa*ra - pb*rb).value

#plt.figure(figsize=(8.4,4.8))

def optimize(epsilon_0: float, epsilon_1: float, beta_0 : float, beta_1 : float,phi,val):
    beta_0 = 2.5
    beta_1 = 0.7
    #phi = phi.value_counts()   # 计数
    #phi.sort_index(inplace=True)
    #D=np.shape(states)[1]
    #K=D+1
    #val_l=200
    #val_u=(-200)

    #val=cp.Variable(K)
    #eps = cp.Variable(1)
    
    #new_s@c_val - out<= eps, out-new_s@c_val <= eps, eps >= 0
    
    N=len(phi)
    #r = cp.Variable(N)
    
    lambda_0= cp.Variable(1)
    lambda_1= cp.Variable(1)
    k=cp.Variable(1)
    ####### lambda_0, lambda_1, k ###############
    #lambda_0 = 0.0
    #lambda_1 = 0.0
    #k = 0.0
    #############################################
    '''abs
    n_iter = 50
    step_size = 0.1
    end_point = 0.001
    
    '''
    u=k-phi
    
    
    f_0_r = cp.sum(cp.maximum(beta_0*u,0))/N - lambda_0*cp.maximum(1-beta_0,0)
    f_1_r = cp.sum(cp.maximum(beta_1*u,0))/N- lambda_1*cp.maximum(1-beta_1,0)
    #for i in range(n_iter):
    prob = k-(lambda_0*epsilon_0+lambda_1*epsilon_1)-(f_0_r+f_1_r)
    cp.Problem(cp.Maximize(prob),[lambda_0>=0,lambda_1>=0,lambda_1<=1,lambda_0<=1,k<=1+min(phi)]).solve(solver = cp.ECOS)
    '''
        o_lambda_0 = lambda_0
        lambda_0 += step_size*(f_0_r-epsilon_0).value
        lambda_0 = max(lambda_0,0)

        o_lambda_1 = lambda_1
        lambda_1 += step_size*(f_1_r-epsilon_1).value
        lambda_1 = max(lambda_1,0)

        o_k = k
        k += step_size*(1-r*avg).value

        if abs(o_lambda_0-lambda_0)<end_point and abs(o_lambda_1-lambda_1)<end_point \
        and abs(o_k-k)<end_point :
            break
    '''
    return  k.value,lambda_0.value,lambda_1.value#,beta.value
def variance(mean,z):
    var=[]
    N=len(z)
    for i in range(N):
        var.append((z[i]-mean)**2)
    final_var=np.sum(var)/N
    return final_var
    
epsilon_PAC=0.01

colors = ['blue','cornflowerblue','deepskyblue','cadetblue']
shapes = ['-', '--', '-.', ':']
for j,sigma in enumerate([0.4]):#0.4,0.6,0.8]):
    data = torch.tensor(torch.load('cartpole_multiframe_sigma_'+str(sigma)+'/best_model.zip_evals_10000.pth'))
    states=torch.tensor(torch.load('cartpole_multiframe_sigma_'+str(sigma)+'/best_model.zipinitial_state.pth'))
    
    probs = torch.tensor(list([(data > i).sum()/10000. for i in range(200)]))
    probs_lb = [ _lower_confidence_bound(int(i*10000.),10000, .05/200) for i in probs]
    print(data[:100])
    print(probs)
    print(probs_lb)
    
    D=np.shape(states)[1]
    K=D+1
    val_l=np.ones(K)*100
    val_u=np.ones(K)*(-100)
    
    vals = []
    var_our = []
    samples=int(2/epsilon_PAC*(np.log(1/0.05)+np.shape(states)[1]))
    #samples=5000
    c_val=cp.Variable(K)
    print(c_val)
    eps = cp.Variable(1)
    out=data[:samples]#/max(data[:samples])
    new_s = torch.tensor(np.append(states[:samples], np.ones((samples, 1)), axis=1))
    cons = [c_val<=200,c_val >= -200,new_s@c_val - out<= eps, out-new_s@c_val <= eps, eps >= 0]
    obj = cp.Minimize(eps)
    prob = cp.Problem(obj, cons)
    prob.solve()
    val = np.array(c_val.value)
    print(c_val.value)
    error=prob.value
    print(error)
    
    #samples=int(2/epsilon_PAC*(np.log(1/0.05)+np.shape(states)[1]))
    
    vals = []
    var_our = []
    data=data.numpy()

    for eps in np.arange(0.00,1.01, 0.01):#1.01
        '''
        data=np.array(data)
        
        lam,k,pro=optimize(eps,data[:samples],states[:samples],val)#
        beta=1
        print('lambda is',lam,'k is',k,'pro is',pro)
        reward=[]
        for i in range(len(data)):
            #reward.append(cp.max(beta*(k-data[i]),0)+ lam*cp.max(1-beta,0))
            reward.append(lam*np.exp((k-data[i])/lam -1))
        ave_reward=np.mean(np.array(reward))
        print('ave_reward is',ave_reward)
        var=variance(ave_reward,np.array(reward))
        print('var is',var)
        R=max(reward)-min(reward)
        N=len(data)
        upper_bound_E= ave_reward+math.sqrt(2*var**2*math.log(3/0.05)/N)+3*R*math.log(3/0.05)/N
        reward_lb=k-lam*eps-upper_bound_E
        var_our.append(reward_lb)
        '''
        '''    
        
        
        k,lambda_0,lambda_1=optimize(epsilon_0,epsilon_1,beta_0,beta_1,data[:1000],val)
        u=k-data
        f_0_r = np.maximum(beta_0*u,0) - lambda_0*np.maximum(1-beta_0,0)
        f_1_r = np.maximum(beta_1*u,0)- lambda_1*np.maximum(1-beta_1,0)
        #for i in range(n_iter):
        reward = (f_0_r+f_1_r)
        ave_reward=np.mean(np.array(reward))
        var=variance(ave_reward,np.array(reward))
        print('var is',var)
        R=max(reward)-min(reward)
        N=len(data)
        upper_bound_E= ave_reward+math.sqrt(2*var**2*math.log(3/0.05)/N)+3*R*math.log(3/0.05)/N
        reward_lb=k-(lambda_0*epsilon_0+lambda_1*epsilon_1)-upper_bound_E
        print('ave_reward is',reward_lb)
        var_our.append(reward_lb)
        
        '''
    
        
        beta_0 = 2.5
        epsilon_0 = get_hs_bound(sigma,norm_bound=eps,beta=beta_0)
        # bound for beta=8
        beta_1 = 0.7
        epsilon_1 = get_hs_bound(sigma,norm_bound=eps,beta=beta_1)
        for i in range(200):
            p = list([data > 199-i])
            print(p)
            result=k,lambda_0,lambda_1=optimize(epsilon_0,epsilon_1,beta_0,beta_1,p[:1000],val)
            u=k-data
            f_0_r = np.maximum(beta_0*u,0) - lambda_0*np.maximum(1-beta_0,0)
            f_1_r = np.maximum(beta_1*u,0)- lambda_1*np.maximum(1-beta_1,0)
            #for i in range(n_iter):
            reward = (f_0_r+f_1_r)
            ave_reward=np.mean(np.array(reward))
            var=variance(ave_reward,np.array(reward))
            R=max(reward)-min(reward)
            N=len(data)
            upper_bound_E= ave_reward+math.sqrt(2*var**2*math.log(3/0.05)/N)+3*R*math.log(3/0.05)/N
            result=k-(lambda_0*epsilon_0+lambda_1*epsilon_1)-upper_bound_E
            print(result)
            if result >=0:
                var_our.append(199-i)
                break
        '''
        for i in range(200):
            pa = probs_lb[(199-i)]
            pb = 1-pa
            pc=0
            result=optimizer(epsilon_0,epsilon_1,beta_0,beta_1,pa,pb,pc)
            if result >=0:
                var_our.append(199-i)
                break
        '''
        avg = 0
        accum = 0
        for i in range(200):
            avg += probs[(199-i)]
            
            accum += get_exact_total(probs_lb[(199-i)],eps, sigma)
            
        #if (eps == 0.01):
            #vals.append(avg)
            #var_our.append(avg)
        vals.append(accum)
    plt.plot(np.arange(0.00,1.01, 0.01),vals,color=colors[j],linestyle=shapes[j],  label= "σ = " + str(sigma)+"baseline")  #"Policy Smoothing: Certified\nLower Bound (σ = " + str(sigma) +')')
    plt.plot(np.arange(0.00,1.01, 0.01),var_our,color='red',linestyle=shapes[j],  label= "σ = " + str(sigma)+"our") 


# attack_mags_nonzero = [0.2,0.4,0.6,0.8,1.0]

# attack_vals =  [torch.tensor(torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_10000.pth')).float().mean().item()]
# attack_sems = [sem(torch.tensor(torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_10000.pth')))]
# for attack_mag in attack_mags_nonzero:
# 	attack_val = None
# 	attack_sem = None
# 	for i,thresh in enumerate([4.,6.,8.,10.]):
# 		cur_val = (torch.tensor( torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_1000_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'.pth')).float().mean().item())
# 		if (attack_val is None or cur_val < attack_val):
# 			attack_val = cur_val
# 			attack_sem = (sem(torch.tensor( torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_1000_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'.pth'))))
# 	attack_vals.append(attack_val)
# 	attack_sems.append(attack_sem)
# attack_mags = [0] + attack_mags_nonzero
# plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, color='red',  linestyle ="--",label="Undefended"  )



# styles = ['-.','--',":"]
# attack_mags_nonzero = [0.2,0.4,0.6,0.8,1.0]
# for j,sigma in enumerate([0.2]):#,0.4]):
# 	attack_vals =  [torch.tensor(torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_10000.pth')).float().mean().item()]
# 	attack_sems = [sem(torch.tensor(torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_10000.pth')))]
# 	for attack_mag in attack_mags_nonzero:
# 		attack_val = None
# 		attack_sem = None
# 		for i,thresh in enumerate([0,1.,2.]):
# 			cur_val = (torch.tensor( torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_1000_smooth_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'_num_smoothing_points_128.pth')).float().mean().item())
# 			if (attack_val is None or cur_val < attack_val):
# 				attack_val = cur_val
# 				attack_sem = (sem(torch.tensor( torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_1000_smooth_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'_num_smoothing_points_128.pth'))))
# 		attack_vals.append(attack_val)
# 		attack_sems.append(attack_sem)
# 	attack_mags = [0] + attack_mags_nonzero
# 	plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, color='blue',  linestyle ="-",label="Policy Smoothing (σ = " + str(sigma) + ')')
plt.legend()
plt.title('(a) Cartpole', fontsize=18)
plt.xlim(0,1.)
plt.xlabel('Perturbation Budget', fontsize=14)
plt.ylim(0,201)
plt.ylabel('Certified Average Score', fontsize=14)
plt.savefig('cartpole_multiframe_certs.png', dpi=400,bbox_inches='tight')
plt.close()