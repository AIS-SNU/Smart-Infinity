#######################################################################################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> LAB=<run1~run8>"
	@echo "      Command to generate the design for specified Target and Device."
	@echo ""
	@echo "  make exe "
	@echo "      Command to generate host."
	@echo ""
	@echo "  make xclbin "
	@echo "      Command to generate hardware platform files(xo,xclbin)."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""
#######################################################################################
TARGET := hw
DEVICE := xilinx_u2_gen3x4_xdma_gc_2_202110_1
LAB := run1
EXECUTABLE := host
XO := krnl_vadd.$(TARGET).$(DEVICE).xo
XCLBIN := krnl_vadd.$(TARGET).$(DEVICE).xclbin
TEST := test.out
RTL_KRNL := ./src/rtl_kernel/rtl_kernel_wizard_0.xo

# Host building global settings
CXXFLAGS := -mf16c -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -Wall -O0 -g -std=c++11 -L$(XILINX_XRT)/lib/ -lOpenCL -pthread -lrt -lstdc++
CXXFLAGS2 := -lOpenCL

# Kernel compiler & linker global settings
#CLFLAGS := -t $(TARGET) --platform $(DEVICE) --config compile.cfg
#LDCLFLAGS := -t $(TARGET) --platform $(DEVICE) --config link.cfg
CLFLAGS := -t $(TARGET) --platform $(DEVICE) -g

LDCLFLAGS := -t $(TARGET) --platform $(DEVICE) --config run.cfg

#LDCLFLAGS += --profile.data:all:all:all
#LDCLFLAGS += --profile.stall:all:all:all
#LDCLFLAGS += --profile.exec:all:all:all

all: $(EXECUTABLE) $(XO) $(XCLBIN) emconfig

exe: $(EXECUTABLE)

xclbin: $(XO) $(XCLBIN)

# Building kernel
ifeq ($(LAB),$(filter $(LAB),run1))
$(XO): ./src/kernel_cpp/adam.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
else ifeq ($(LAB),$(filter $(LAB),run2))
$(XO): ./src/kernel_cpp/adam_topk.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
else ifeq ($(LAB),$(filter $(LAB),run2))
$(XO): ./src/kernel_cpp/adagrad.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
else ifeq ($(LAB),$(filter $(LAB),run2))
$(XO): ./src/kernel_cpp/aagrad_topk.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
else ifeq ($(LAB),$(filter $(LAB),run2))
$(XO): ./src/kernel_cpp/sgd.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
else ifeq ($(LAB),$(filter $(LAB),run2))
$(XO): ./src/kernel_cpp/sgd_topk.cpp
	v++ $(CLFLAGS) -c -k krnl_vadd -I'$(<D)' -o'$@' '$<'
endif


$(XCLBIN): krnl_vadd.$(TARGET).$(DEVICE).xo 
	v++ $(LDCLFLAGS) -l -o'$@' $(+)

# Building Host
ifeq ($(LAB),$(filter $(LAB),run1))
$(EXECUTABLE): ./src/host/host_step_adam.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
else ifeq ($(LAB),$(filter $(LAB),run2))
$(EXECUTABLE): ./src/host/host_step_adam_topk.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
else ifeq ($(LAB),$(filter $(LAB),run3))
$(EXECUTABLE): ./src/host/host_step_adagrad.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
else ifeq ($(LAB),$(filter $(LAB),run4))
$(EXECUTABLE): ./src/host/host_step_adagrad_topk.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
else ifeq ($(LAB),$(filter $(LAB),run5))
$(EXECUTABLE): ./src/host/host_step_sgd.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
else ifeq ($(LAB),$(filter $(LAB),run6))
$(EXECUTABLE): ./src/host/host_step_sgd_topk.cpp
	g++ $(CXXFLAGS) -o '$@' '$<' $(CXXFLAGS2)
endif


.PHONY: emconfig
emconfig:
	emconfigutil --platform $(DEVICE)

.PHONY: run
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(XCLBIN)
else
	./$(EXECUTABLE) $(XCLBIN)
endif


# Cleaning stuff
.PHONY: clean

RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*}
	-$(RMDIR) TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) *.protoinst _v++_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.ltx

cleanall: clean
	-$(RMDIR) $(XCLBIN) *.xo *.xclbin* *.wcfg *.wdb *.csv *.compile_summary *.run_summary *.ltx
	-$(RMDIR) _x* .run/

