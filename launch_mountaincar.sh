#!/usr/bin/env bash

# nohup python run_mountaincar_cli.py --use-memory false --model-name llama3.1-8b --memory-env vanilla --use-global-verifier false --use-api false --cuda-visible-devices 1 > nohup_mountaincar_log_llama3.1-8b_nomemory.out 2>&1 & echo "PID=$!"
# nohup python run_mountaincar_cli.py --use-memory true --model-name llama3.1-8b --memory-env voyager --use-global-verifier true --use-api false --cuda-visible-devices 2 > nohup_mountaincar_log_llama3.1-8b_voyager_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_mountaincar_cli.py --use-memory true --model-name llama3.1-8b --memory-env generative --use-global-verifier true --use-api false --cuda-visible-devices 3 > nohup_mountaincar_log_llama3.1-8b_generative_global_verifier.out 2>&1 & echo "PID=$!"
# nohup python run_mountaincar_cli.py --use-memory true --model-name llama3.1-8b --memory-env memorybank --use-global-verifier true --use-api false --cuda-visible-devices 4 > nohup_mountaincar_log_llama3.1-8b_memorybank_global_verifier.out 2>&1 & echo "PID=$!"

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

