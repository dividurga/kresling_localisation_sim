# magpylib >=5 uses strict SI units internally (meters, Tesla, A/m, A*m^2) --
# NOT the mm/mT convention of older versions. This project's public
# interfaces stay in mm/mT (matching how the papers report everything, and
# matching the physical intuition of a few-mm robot), and convert to/from SI
# only at the magpylib boundary. Never pass raw mm/mT values into magpy.*
# objects directly anywhere else in this codebase -- go through these.
MM_TO_M = 1e-3
M_TO_MM = 1e3
MT_TO_T = 1e-3
T_TO_MT = 1e3
