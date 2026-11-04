# Reading List

## Numerical integration: Euler vs. RK4 (and symplectic methods)

Context: `sablas/envs/env_satellite.py` originally integrated the satellite's
orbit with explicit Euler, which spiraled a 540 km circular orbit out to
1400+ km within one revolution. Switching to RK4 held the same orbit to mm
accuracy over 5 revolutions. These cover why, and where RK4 itself still
falls short over long integrations.

- Wisdom, J. & Holman, M. (1991), *Symplectic Maps for the N-Body Problem*,
  The Astronomical Journal, 102(4), 1528-1538.
  [NASA ADS/NTRS](https://ntrs.nasa.gov/citations/19910069505)
  The foundational paper on symplectic integrators for nearly-Keplerian
  systems — explains why energy-preserving (symplectic) methods beat
  higher-order non-symplectic ones like RK4 over very long orbital
  integrations.

- Hairer, E., McLachlan, R. & Razakarivony, A. (2008), *Splitting and
  composition methods in the numerical integration of differential
  equations*. [arXiv:0812.0377](https://arxiv.org/abs/0812.0377)
  Survey of splitting/symplectic composition methods, with the Euler vs.
  higher-order vs. symplectic trade-off spelled out for Hamiltonian systems.

- Kokubu, K. et al., *Time-symmetric integration in astrophysics*.
  [arXiv:1708.07266](https://arxiv.org/abs/1708.07266)
  Compares integrator families (including RK4) on long-term orbital/N-body
  problems and shows RK4's energy error still grows secularly, just far
  slower than Euler's.

- *Numerical Integration Techniques in Orbital Mechanics Applications*.
  [ResearchGate](https://www.researchgate.net/publication/281108779_Numerical_Integration_Techniques_in_Orbital_Mechanics_Applications)
  Applied comparison of integrator choices (Euler, RK families, etc.)
  specifically for orbit propagation — directly relevant to `leo_sim.py` /
  `env_satellite.py`.

- *A Comparative Study on Numerical Solutions of Initial Value Problems
  (IVP) for Ordinary Differential Equations (ODE) with Euler and Runge-Kutta
  Methods*, American Journal of Computational Mathematics (2015).
  [SCIRP](https://www.scirp.org/journal/paperinformation?paperid=59961)
  General, worked-example comparison of Euler vs. RK4 accuracy/error
  order — good first read before the orbital-mechanics-specific papers above.

- Rycroft, C., *AM225: Symplectic Integration Methods* (lecture notes,
  Harvard). [PDF](https://people.math.wisc.edu/~chr/am225/notes/am225_symplectic.pdf)
  Not a paper, but a concise derivation of why forward Euler drifts
  outward, backward Euler drifts inward, and symplectic Euler/midpoint
  methods don't drift at all — the clearest short explanation of the effect
  observed in `env_satellite.py`.
