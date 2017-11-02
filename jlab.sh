mkdir /root/.jupyter/
wget https://raw.githubusercontent.com/ivukotic/rl_examples/master/jupyter_notebook_config.py
mv jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
export SHELL=/bin/bash

jupyter lab --allow-root
