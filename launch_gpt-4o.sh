#!/usr/bin/env bash
# mountaincar
# gpt-4o
nohup python run_mountaincar_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_mountaincar_log_gpt-4o_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_mountaincar_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier false > nohup_mountaincar_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier false > nohup_mountaincar_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier false > nohup_mountaincar_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier true > nohup_mountaincar_log_gpt-4o_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier true > nohup_mountaincar_log_gpt-4o_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier true > nohup_mountaincar_log_gpt-4o_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier true > nohup_mountaincar_log_gpt-4o_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# # webshop
# # gpt-4o
# nohup python run_webshop_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_webshop_log_gpt-4o_nomemory.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_webshop_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier false > nohup_webshop_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier false > nohup_webshop_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier false > nohup_webshop_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
# # use global verifier
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier true > nohup_webshop_log_gpt-4o_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier true > nohup_webshop_log_gpt-4o_voyager_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier true > nohup_webshop_log_gpt-4o_generative_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier true > nohup_webshop_log_gpt-4o_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# # webshop_hidden
# # gpt-4o
# nohup python run_webshop_hidden_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_webshop_hidden_log_gpt-4o_nomemory.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_webshop_hidden_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier false > nohup_webshop_hidden_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier false > nohup_webshop_hidden_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier false > nohup_webshop_hidden_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
# # use global verifier
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier true > nohup_webshop_hidden_log_gpt-4o_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier true > nohup_webshop_hidden_log_gpt-4o_voyager_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier true > nohup_webshop_hidden_log_gpt-4o_generative_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_webshop_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier true > nohup_webshop_hidden_log_gpt-4o_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# frozenlake
# gpt-4o
# nohup python run_frozenlake_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_frozenlake_log_gpt-4o_nomemory.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_frozenlake_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier false > nohup_frozenlake_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier false > nohup_frozenlake_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier false > nohup_frozenlake_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
# # use global verifier
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier true > nohup_frozenlake_log_gpt-4o_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier true > nohup_frozenlake_log_gpt-4o_voyager_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier true > nohup_frozenlake_log_gpt-4o_generative_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier true > nohup_frozenlake_log_gpt-4o_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# # frozenlake_hidden
# # gpt-4o
# nohup python run_frozenlake_hidden_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_frozenlake_hidden_log_gpt-4o_nomemory.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier false > nohup_frozenlake_hidden_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier false > nohup_frozenlake_hidden_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier false > nohup_frozenlake_hidden_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier false > nohup_frozenlake_hidden_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
# # use global verifier
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla --use-global-verifier true > nohup_frozenlake_hidden_log_gpt-4o_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env voyager --use-global-verifier true > nohup_frozenlake_hidden_log_gpt-4o_voyager_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env generative --use-global-verifier true > nohup_frozenlake_hidden_log_gpt-4o_generative_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank --use-global-verifier true > nohup_frozenlake_hidden_log_gpt-4o_memorybank_global_verifier.out 2>&1 & echo "PID=$!"
