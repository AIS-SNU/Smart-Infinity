# \[HPCA'24\] Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System  

We identify that moving parameter updates to the storage side removes most of the storage traffic. In addition, we propose an efficient data transfer handler structure to address the system integration issues for Smart-Infinity. 
The handler allows overlapping data transfers with fixed memory consumption by reusing the device buffer. 
Lastly, we propose accelerator-assisted gradient compression/decompression to enhance the scalability of Smart-Infinity. 
As a result, Smart-Infinity achieves a significant speedup compared to the baseline. 

## Motivation
The Large Language Models (LLMs) is mainly driven by the increase in the number of parameters. 
This has led to substantial memory capacity requirements, necessitating the use of dozens of GPUs just to meet the capacity. 
One popular solution to this is storage-offloaded training, which uses host memory and storage as an extended memory hierarchy. 
However, this obviously comes at the cost of storage bandwidth bottleneck because storage devices have orders of magnitude lower bandwidth compared to that of GPU device memories. 
Our work, Smart-Infinity, addresses the storage bandwidth bottleneck of storage-offloaded LLM training using near-storage processing devices on a real system. 

---

## 📰 News
- 🐝 [24/03] Smart-Infinity received **Best-paper honorable methioned** at **HPCA'24**.


## Content
- [Codebases](#codebases)
- [Setups](#setups)
- [Overall Steps](#overall-steps)
- [Some Tips](#some-tips)
- [Roadmap](#roadmap)

## Codebases

- DeepSpeedExamples
(https://github.com/microsoft/DeepSpeedExamples)

- DeepSpeed
(https://github.com/microsoft/DeepSpeed)

- Xilinx Vitis Tutorials
(https://docs.xilinx.com/v/u/en-US/UG1605-vitis-tutorials)

## Setups

### Hardware
NVIDIA RTX A100-GPU (40GB) server with 2x 48-core Intel Xeon(R) Gold 6342 CPU and 32\*32GB of DDR4 RAM, Samsung SmartSSD.

### Operating System
Ubutun 20.04 with Linux kernel 5.4.0-156

### Software
CUDA 11.6, PyTorch 1.12.1

## Overall Steps

### 1. See `setup` 
The `setup` folder contains how to install mandatory libararies and configurations of our work. The setup can be changed when the experimental setup is different. 

### 2. See ```hls_smartInfinity``` 
Our main contribution is performing parameter update in CSD, so device binary file for FPGA is needed. `hls_smartInfinity` folder contains how to make binary file for parameter update, the result of the process is single binary file in the designated folder.

### 3. Run training
After all the setups are prepared, you can get started with ```run_smartinfinity.sh``` as a entry file of our work.

## Some Tips

We recommend not to use docker because P2P direct communication feature of Xilinx FPGA seems to not properly work in container.

See `DeepSpeedExample/example/ds_zero_stage_infinity-nvme.json` for important hyperparameters for using SmartInfinity. 


