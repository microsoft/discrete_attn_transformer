echo on
call conda update -y -n base -c defaults conda
call conda create -y -n icl python=3.10
call conda activate icl

call conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
call pip install -r requirements.txt

call conda develop .

