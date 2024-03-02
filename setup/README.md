# Setup process for SmartInfinity

We use WikiText-103 dataset for this work, refer to Megatron-LM github (https://github.com/NVIDIA/Megatron-LM#gpt-text-generation), but different datasets may not significant effect to performance.

## Mount SSDs to designated name

In our implementation, mounted folders have to follow our name convention to recognize each CSDs.

See attached FPGA order using `xbutil examine`, and figure out each PCIe address of FPGA.

And mount the corresponding SSD to `smartssd{FPGA_index}`.

For example, if the attached FPGA order in `xbutil examine` is 3, then mount corresponding SSD to `smartssd3`.

## Replace installed deepspeed libaray to customized deepspeed for SmartInfinity

See `setup_deepspeed.sh` file to replce original deepspeed liabary and some mandatory libraries (apex: https://github.com/NVIDIA/apex, regex)
To use (NVME) aio, use `DS_BUILD_OPS=1`.

# Some scripts for setups

`mdadm.sh`: This script helps to automatically create software raid setup for RAID baseline. For only smartinfinity, you do not need to run this script. `raid{#SSDs}` are used for the name convetion.

`nvme_speed_test.py` : If you want to check nvme SSD read/write performance, you can use this file for measurement.

`cleans_ssd.sh` : Erase all the data in SSD.

*We strongly recommend to use conda environment* for OpenCL features and P2P feature of SAMSUNG SmartSSD.


