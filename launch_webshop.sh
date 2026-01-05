#!/usr/bin/env bash

nohup python run_webshop_cli.py --use-memory false --model-name llama3.1-8b --memory-env vanilla --use-api false --cuda-visible-devices 1 > nohup_webshop_log_llama3.1-8b_plain.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name llama3.1-8b --memory-env vanilla --use-api false --cuda-visible-devices 2 > nohup_webshop_log_llama3.1-8b_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name llama3.1-8b --memory-env voyager --use-api false --cuda-visible-devices 3 > nohup_webshop_log_llama3.1-8b_voyager.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name llama3.1-8b --memory-env generative --use-api false --cuda-visible-devices 5 > nohup_webshop_log_llama3.1-8b_generative.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name llama3.1-8b --memory-env memorybank --use-api false --cuda-visible-devices 6 > nohup_webshop_log_llama3.1-8b_memorybank.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name llama3.1-8b --memory-env glove --use-api false --cuda-visible-devices 7 > nohup_webshop_log_llama3.1-8b_glove.out 2>&1 & echo "PID=$!"

nohup python run_webshop_cli.py --use-memory false --model-name gpt-4o --memory-env vanilla > nohup_webshop_log_gpt-4o_plain.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env vanilla > nohup_webshop_log_gpt-4o_vanilla.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env voyager > nohup_webshop_log_gpt-4o_voyager.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env generative > nohup_webshop_log_gpt-4o_generative.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env memorybank > nohup_webshop_log_gpt-4o_memorybank.out 2>&1 & echo "PID=$!"
nohup python run_webshop_cli.py --use-memory true --model-name gpt-4o --memory-env glove > nohup_webshop_log_gpt-4o_glove.out 2>&1 & echo "PID=$!"