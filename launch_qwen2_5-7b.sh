#!/usr/bin/env bash
# mountaincar
# qwen2.5-7b-instruct
nohup python run_mountaincar_cli.py --use-memory false --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_mountaincar_log_qwen2.5-7b-instruct_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_mountaincar_log_qwen2.5-7b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier false > nohup_mountaincar_log_qwen2.5-7b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier false > nohup_mountaincar_log_qwen2.5-7b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier false > nohup_mountaincar_log_qwen2.5-7b-instruct_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier true > nohup_mountaincar_log_qwen2.5-7b-instruct_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier true > nohup_mountaincar_log_qwen2.5-7b-instruct_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier true > nohup_mountaincar_log_qwen2.5-7b-instruct_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_mountaincar_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier true > nohup_mountaincar_log_qwen2.5-7b-instruct_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# webshop
# qwen2.5-7b-instruct
nohup python run_webshop_cli.py --use-memory false --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_webshop_log_qwen2.5-7b-instruct_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_webshop_log_qwen2.5-7b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier false > nohup_webshop_log_qwen2.5-7b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier false > nohup_webshop_log_qwen2.5-7b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier false > nohup_webshop_log_qwen2.5-7b-instruct_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier true > nohup_webshop_log_qwen2.5-7b-instruct_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier true > nohup_webshop_log_qwen2.5-7b-instruct_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier true > nohup_webshop_log_qwen2.5-7b-instruct_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier true > nohup_webshop_log_qwen2.5-7b-instruct_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# webshop_hidden
# qwen2.5-7b-instruct
nohup python run_webshop_hidden_cli.py --use-memory false --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_webshop_hidden_log_qwen2.5-7b-instruct_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_webshop_hidden_log_qwen2.5-7b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier false > nohup_webshop_hidden_log_qwen2.5-7b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier false > nohup_webshop_hidden_log_qwen2.5-7b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier false > nohup_webshop_hidden_log_qwen2.5-7b-instruct_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier true > nohup_webshop_hidden_log_qwen2.5-7b-instruct_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier true > nohup_webshop_hidden_log_qwen2.5-7b-instruct_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier true > nohup_webshop_hidden_log_qwen2.5-7b-instruct_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_webshop_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier true > nohup_webshop_hidden_log_qwen2.5-7b-instruct_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# frozenlake
# qwen2.5-7b-instruct
nohup python run_frozenlake_cli.py --use-memory false --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_frozenlake_log_qwen2.5-7b-instruct_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_frozenlake_log_qwen2.5-7b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier false > nohup_frozenlake_log_qwen2.5-7b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier false > nohup_frozenlake_log_qwen2.5-7b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier false > nohup_frozenlake_log_qwen2.5-7b-instruct_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier true > nohup_frozenlake_log_qwen2.5-7b-instruct_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier true > nohup_frozenlake_log_qwen2.5-7b-instruct_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier true > nohup_frozenlake_log_qwen2.5-7b-instruct_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier true > nohup_frozenlake_log_qwen2.5-7b-instruct_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

# frozenlake_hidden
# qwen2.5-7b-instruct
nohup python run_frozenlake_hidden_cli.py --use-memory false --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_nomemory.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier false > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier false > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier false > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier false > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_memorybank.out 2>&1 & echo "PID=$!"
# use global verifier
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env vanilla --use-global-verifier true > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_vanilla_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env voyager --use-global-verifier true > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_voyager_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env generative --use-global-verifier true > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_generative_global_verifier.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name qwen2.5-7b-instruct --memory-env memorybank --use-global-verifier true > nohup_frozenlake_hidden_log_qwen2.5-7b-instruct_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

