#!/usr/bin/env bash

nohup python run_frozenlake_hidden_cli.py --use-memory false --model-name llama-3.3-70b-instruct --memory-env vanilla > nohup_log_hidden_llama-3.3-70b-instruct_plain.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name llama-3.3-70b-instruct --memory-env vanilla > nohup_log_hidden_llama-3.3-70b-instruct_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name llama-3.3-70b-instruct --memory-env voyager > nohup_log_hidden_llama-3.3-70b-instruct_voyager.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name llama-3.3-70b-instruct --memory-env generative > nohup_log_hidden_llama-3.3-70b-instruct_generative.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name llama-3.3-70b-instruct --memory-env memorybank > nohup_log_hidden_llama-3.3-70b-instruct_memorybank.out 2>&1 & echo "PID=$!"
nohup python run_frozenlake_hidden_cli.py --use-memory true --model-name llama-3.3-70b-instruct --memory-env glove > nohup_log_hidden_llama-3.3-70b-instruct_glove.out 2>&1 & echo "PID=$!"
