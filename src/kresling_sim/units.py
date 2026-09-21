# magpylib >=5 uses strict SI internally (m, T, A/m, A*m^2), not the mm/mT
# of older versions. We keep public interfaces in mm/mT (matches the papers
# and the physical intuition of a few-mm robot) and convert only at the
# magpylib boundary -- never pass raw mm/mT into magpy.* objects directly.
MM_TO_M = 1e-3
M_TO_MM = 1e3
MT_TO_T = 1e-3
T_TO_MT = 1e3
