find ./log -type f ! -name 'placeholder' -delete
find ./storage -type f ! -name 'placeholder' -delete
find . -maxdepth 1 -type f -name 'nohup*' -delete
find . -maxdepth 1 -type d -name 'log_*' -exec rm -rf {} +