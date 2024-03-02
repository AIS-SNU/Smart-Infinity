# Smart-Infinity HLS files

Our code base of this HLS project is https://docs.xilinx.com/v/u/en-US/UG1605-vitis-tutorials 

## Step 0 : Prerequisite
- Install Vitis HLS 2023.1
- Install xrt (2.12.427)
- See https://xilinx.github.io/XRT/2022.1/html/p2p.html, if all the setups are done for P2P direct communication.
- For smartSSD firmware settings, See https://www.xilinx.com/content/dam/xilinx/support/documents/boards_and_kits/accelerator-cards/1_3/ug1382-smartssd-csd.pdf

## Step 1 : Generate binary file
See `MakeFile` for implementations of other types optimizers.
We provide our implementation files for SGD and Adagrad.

``` bash
make xclbin LAB=run1 #Adam only
make xclbin LAB=run2 #SmartComp topk compression + Adam
```
After compilation, you can see the generated `*.xclbin` file.

If you want to apply your own optimizers or compression algorthm, 
see `src/kernel_cpp/` directory.

## Step 2 : Sanity Checker for binary file 

You can see `src/host` files for providing functionality check.

- For w/o topk compression,
``` bash
make host LAB=run1 # Adam only
./host
```

- For w/ topk compression,
``` bash
make host LAB=run2 #SmartComp Topk compression + Adam
./host
```

## Step 3 : Copy to appropriate directory for SmartInfinity

Default path for Smart-Infnity is `($HOME)/bins/adam.xclbin` for adam only binary file and `($HOME)/bins/topk_adam.xclbin`.


## Some Guidances for Data Types

We provide our experiences for types of gradients and updated parameters.
We think there are pros and cons in using different types for those.

- Updated parameters

If the parameters are sent in FP16, FPGAs have to convert master weights into FP16, which menas anoter writing is needed. So, the addtional overhead results in the degradation of throughput. But, this can reduce traffic through shared PCIe lanes, so this implementation provides further speedup when #SmartSSD is larger.

- Gradients

If the gradients are sent in FP16 from host, the total traffic through shared PCIe lanes can be reduced. However, in finetuning scenarios, a technique called as gradient accumulation is widely used for training. So, overflow is more frequently occurred with accumulation of gradient in FP16, so more learng rate scaling might occur.
Additionally, if you use gradients in FP16 the effect of our gradient compression can be reduced.

In our paper, the implementations and evalution results are conducted with using FP32 for both parameters and gradients.

We recommend to use setup for sending paremeter to host in FP16, and receiving gradients in FP32, which can show the highest performance with no limtations.