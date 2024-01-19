# Medical_Chatbot_LLM

This talks about creating a ChatBot for answering question regarding Medicine.

We are using [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M) (a Diverse Herd of Distilled Models from Large-Scale Instructions) a large language model is a fine-tuned version of t5-large on LaMini-instruction dataset that contains [2.58M](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) samples for instruction fine-tuning.

To answer questions on Medicine we have to train the model with text pretaining to encylopedia for medicines etc.

So for that we will using pdf books: <\br>
* [The GALE ENCYCLOPEDIA of MEDICINE](http://ndl.ethernet.edu.et/bitstream/123456789/69488/1/60.pdf.pdf)

![Alt text](images/llm_design.drawio.png)

login as root
```
conda create -n deeplearning101 python=3.10

```

activate deeplearning101 in root user

```
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
```

### Download GALE ENCYCLOPEDIA of MEDICINE
```
    mkdir -p docs
   curl -O http://ndl.ethernet.edu.et/bitstream/123456789/69488/1/60.pdf.pdf
```

### Clone LaMini-T5-738M from HuggingFace

Install git lfs before cloning, for ubuntu

```
sudo apt-get install git-lfs
```
to clone large files use git-lfs
```
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
```

### Running Gradio on FastAPI using Gunicorn

```
gunicorn run:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 --daemon
```
