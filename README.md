# ReCePS
Our paper Reward Certification for Policy Smoothed Reinforcement Learning is accepted by AAAI 2024

https://ojs.aaai.org/index.php/AAAI/article/view/30139

The appendix is attached

# Train the policy smoothing model
The model training code and attack method are released by the paper "Policy Smoothing for Provably Robust Reinforcement Learning"

python cartpole_multiframe_train.py  --sigma 0.1 (std. dev 0.2)

python freeway_train.py  --sigma 0.00  Trains an undefended model

# Attack the policy smoothing model and undefended model

python3 action_attack_cartpole.py --p 0.16 --attack_eps 40 --q_threshold 0.0 --checkpoint cartpole_multiframe_sigma_0.0/best_model

python3 cartpole_simple_attack.py --checkpoint  cartpole_simple_sigma_0.0/best_model.zip


# Experiment Result

<img width="497" alt="Screenshot 2024-12-16 at 16 35 58" src="https://github.com/user-attachments/assets/c92c5057-c51b-47f2-b2dc-a5a23046e770" />
