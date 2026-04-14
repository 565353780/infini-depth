pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install xformers==0.0.29.post3 \
  --index-url https://download.pytorch.org/whl/cu124

pip install huggingface-hub timm lpips tyro pandas einops h5py \
  omegaconf dacite hydra-core scikit-learn opencv-python \
  scikit-image imageio imageio[ffmpeg] matplotlib plyfile \
  open3d OpenEXR viser e3nn roma torchmetrics fvcore iopath \
  lpips pytorch_lightning wandb tensorboard rich tqdm colorama \
  termcolor jaxtyping beartype bidict tabulate svg.py sk-video \
  ipdb pdbr ffmpeg-python colorspacious regex onnxruntime gradio \
  huggingface_hub plotly spaces trimesh

pip install numpy <2.0
pip install moviepy==1.0.3
pip install protobuf==3.20.2
