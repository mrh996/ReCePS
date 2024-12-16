# ReCePS
Our paper Reward Certification for Policy Smoothed Reinforcement Learning is accepted by AAAI 2024

https://ojs.aaai.org/index.php/AAAI/article/view/30139

The appendix is attached

# Train the policy smoothing model
Some model training codes and attack codes are from the paper "Policy Smoothing for Provably Robust Reinforcement Learning"

python cartpole_multiframe_train.py  --sigma 0.1 (std. dev 0.2)

python freeway_train.py  --sigma 0.00  Trains an undefended model

# Attack the policy smoothing model and undefended model


Attack smoothed model

python3 mountainattackl1.py --sigma 0.4  --attack_eps 2 --norm_coeff 0.001 --checkpoint  mountain_car_sigma_0.4/best_model

Attack undefended model

python3 cartpole_simple_attack.py --checkpoint  cartpole_simple_sigma_0.0/best_model.zip

python3 freeway_attack_smooth.py --sigma 2.55 --attack_eps 102.0 --q_threshold 0.0 --checkpoint freeway_sigma_2.55_1/best_model

# Test model (for generating certificates):

python3 cartpole_multiframe_test_l0.py  --sigma 0.0 --checkpoint cartpole_multiframe_ef_0.16_sigma_0.0/best_model.zip 

python3 cartpole_multiframe_test.py  --sigma 0.1 --checkpoint cartpole_multiframe_sigma_0.1/best_model

python3 freeway_testl0.py  --sigma 0.0 --checkpoint freeway_ef_0.2/best_model


# Experiment Result

<img width="497" alt="Screenshot 2024-12-16 at 16 35 58" src="https://github.com/user-attachments/assets/c92c5057-c51b-47f2-b2dc-a5a23046e770" />
