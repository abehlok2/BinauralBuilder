from binauralbuilder_core.utils.numba_status import configure_numba

# Ensure we log Numba availability and install a stub before importing synth modules.
configure_numba()
