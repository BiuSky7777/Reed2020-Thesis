# Reed2020-Thesis

## Codes for Quantum Models (Section 5)
`Schmidt_Decomposition.py` is the code for Schmidt decomposition of a general matrix of dimension 2<sup>n</sup> × 2<sup>n</sup>.

### XXZ Model

`XXZ_Schmidt_Decomposition.py` is the code for Schmidt decomposition related calculation like entropy and Schmidt coefficients of XXZ model.

`XXZ_Classical_Hamiltonian.py` is the code to construct XXZ Hamiltonian using Algorithm 3 in section 4.

`XXZ_Computation.py` is the code to generate data for the plots in the thesis.

`XXZ_Computation_Plot.nb` is the Mathemaica code that takes data generated by to `XXZ_Computation.py` to make plots used in the thesis.

### XXZ Variant Model

`XXZ_Schmidt_Decomposition.py`, `XXZ_Computation.py`,`XXZ_Computation_Plot.nb` are also used for generating plots for XXZ Variant model.

`XXZ_Variant_Hamiltonian.py` is the code to construct XXZ Variant Hamiltonians using Algorithm 1 in Section 4.

### AKLT Model

`AKLT_Hamiltonian.py` is the code to construct AKLT Hamiltonian using Algorithm 1 in section 4, and computes entanglement entropy and Schmidt coefficients.

(Note that we do not use `Schmidt_Decomposition.py` and `XXZ_Schmidt_Decomposition.py` because AKLT Hamiltonian is of dimension 3<sup>n</sup> × 3<sup>n</sup>.)

`AKLT_Computation.py` is the code to generate data for the plots in the thesis.

`AKLT_Computation_Plot.nb` is the Mathemaica code that takes data generated by to `AKLT_Computation.py` to make plots used in the thesis.

## Codes for QAGSP Evaluation (Section 6)

`QAGSP` is the code for general QAGSP construction described in Section 6.1.

`XXZ_Classical_QAGSP.py` is the code to investigate entanglement entropy and Schmidt coefficients change with a QAGSP being kept applying on a random product state.

`XXZ_Classical_QAGSP_Plot.nb` is the Mathemaica code that takes data generated by to `XXZ_Classical_QAGSP.py` to make plots used in the thesis.