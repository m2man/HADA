import os
import sys

bl_m = os.path.abspath('./Baseline_m')
bl_m_e = os.path.abspath('./Baseline_m_extend')
lifu_m = os.path.abspath('./Lifu_m')
lifu_m_e = os.path.abspath('./Lifu_m_extend')

libs = [bl_m, bl_m_e, lifu_m, lifu_m_e]
for lib in libs:
    if lib not in sys.path:
        sys.path.append(lib)