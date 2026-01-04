#!/usr/bin/env bash

nohup python run_frozenlake_hidden_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla > nohup_log_hidden_gpt-4o_plain.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla > nohup_log_hidden_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env voyager > nohup_log_hidden_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env generative > nohup_log_hidden_gpt-4o_generative.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank > nohup_log_hidden_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name gpt-4o --memory-env glove > nohup_log_hidden_gpt-4o_glove.out 2>&1 & echo "PID=$!"
