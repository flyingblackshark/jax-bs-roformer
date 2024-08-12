pip install --upgrade pip setuptools
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/boris-kuz/jaxloudnorm
pip install -r requirements.txt