echo "Launcher Script(emulation mode only)"
echo "XILINX_SDX PATH = ${XILINX_SDX}"
echo "Setting Emulation Config..."
emconfigutil --platform @platform@ --nd 1
export XCL_EMULATION_MODE=sw_emu
./MyHostExecutable
