
import scipy.special
from scipy.stats import norm, binom_test,sem
from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

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

plt.figure(figsize=(8.4,4.8))


# colors = ['blue','cornflowerblue','deepskyblue','cadetblue']
# shapes = ['-', '--', '-.', ':']
# for j,sigma in enumerate([0.2,0.4]):#,0.6,0.8]):
# 	data = torch.tensor(torch.load('cartpole_multiframe_sigma_'+str(sigma)+'/best_model.zip_evals_10000.pth'))
# 	probs = torch.tensor(list([(data > i).sum()/10000. for i in range(200)]))
# 	probs_lb = [ _lower_confidence_bound(int(i*10000.),10000, .05/200) for i in probs]
# 	print(probs)
# 	print(probs_lb)

# 	vals = []
# 	for eps in np.arange(0.01,1.01, 0.01):
# 		avg = 0
# 		accum = 0
# 		for i in range(200):
# 			avg += probs[(199-i)]
# 			accum += get_exact_total(probs_lb[(199-i)],eps, sigma)
# 		if (eps == 0.01):
# 			vals.append(avg)
# 		vals.append(accum)
# 	plt.plot(np.arange(0.00,1.01, 0.01),vals,color=colors[j],linestyle=shapes[j],  label= "σ = " + str(sigma))  #"Policy Smoothing: Certified\nLower Bound (σ = " + str(sigma) +')')


attack_mags_nonzero = [0.2,0.4,0.6,0.8,1.0]

attack_vals =  [torch.tensor(torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_10000.pth')).float().mean().item()]
attack_sems = [sem(torch.tensor(torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_10000.pth')))]
for attack_mag in attack_mags_nonzero:
	attack_val = None
	attack_sem = None
	for i,thresh in enumerate([4.,6.,8.,10.]):
		cur_val = (torch.tensor( torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_1000_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'.pth')).float().mean().item())
		if (attack_val is None or cur_val < attack_val):
			attack_val = cur_val
			attack_sem = (sem(torch.tensor( torch.load('cartpole_multiframe_sigma_0.0/best_model.zip_evals_1000_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'.pth'))))
	attack_vals.append(attack_val)
	attack_sems.append(attack_sem)
attack_mags = [0] + attack_mags_nonzero
plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, color='red',  linestyle ="--",label="Undefended"  )



styles = ['-.','--',":"]
attack_mags_nonzero = [0.2,0.4,0.6,0.8,1.0]
for j,sigma in enumerate([0.2]):#,0.4]):
	attack_vals =  [torch.tensor(torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_10000.pth')).float().mean().item()]
	attack_sems = [sem(torch.tensor(torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_10000.pth')))]
	for attack_mag in attack_mags_nonzero:
		attack_val = None
		attack_sem = None
		for i,thresh in enumerate([0,1.,2.]):
			cur_val = (torch.tensor( torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_1000_smooth_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'_num_smoothing_points_128.pth')).float().mean().item())
			if (attack_val is None or cur_val < attack_val):
				attack_val = cur_val
				attack_sem = (sem(torch.tensor( torch.load('cartpole_multiframe_sigma_0'+str(sigma)+'/best_model.zip_evals_1000_smooth_eps_'+str(attack_mag)+'_attack_step_count_multiplier_2_attack_step_0.01_threshold_'+str(thresh)+'_num_smoothing_points_128.pth'))))
		attack_vals.append(attack_val)
		attack_sems.append(attack_sem)
	attack_mags = [0] + attack_mags_nonzero
	plt.errorbar([x for x in attack_mags],attack_vals,  yerr= attack_sems, color='blue',  linestyle ="-",label="Policy Smoothing (σ = " + str(sigma) + ')')



plt.legend()
plt.title('(a) Cartpole', fontsize=18)
plt.xlim(0,1.)
plt.xlabel('Perturbation Budget', fontsize=14)
plt.ylim(0,201)
plt.ylabel('Average Score', fontsize=14)
plt.savefig('cartpole_multiframe_attacks.png', dpi=400,bbox_inches='tight')
