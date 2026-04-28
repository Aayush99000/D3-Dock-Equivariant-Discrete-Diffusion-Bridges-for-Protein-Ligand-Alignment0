# D3-Dock: Surface-Aware Equivariant Discrete Diffusion for Protein-Ligand Alignment

**D3-Dock** is a next-generation generative framework for protein-ligand docking. While previous models often suffer from **structural dissonance**—a mismatch between discrete chemical bonds and continuous 3D coordinates—D3-Dock uses a **Neural Schrödinger Bridge** and **Discrete Denoising Diffusion (D3PM)** to "guardrail" the folding process.

![D3-Dock Architecture](architecture.png)

## Key Innovations

- **Surface-Aware Physical Guardrails:** Integrates a Point Cloud Transformer to encode the protein "skin." By utilizing a **Signed Distance Function (SDF)**, the model prevents "physical hallucinations" where ligands overlap with protein atoms.
- **Discrete-Led Refinement:** Employs a transition matrix $Q_t$ to evolve atom and bond types discretely, ensuring $100\%$ chemical valency throughout the diffusion process.
- **$E(3)$-Equivariant Architecture:** The underlying Graph Neural Network is rotationally and translationally invariant, respecting the fundamental physics of 3D space.
- **Neural Schrödinger Bridge:** Finds the entropy-optimal transport path between random noise and the docked state, significantly improving sampling efficiency over standard diffusion.

## 🧬 Technical Deep Dive: The Neural Schrödinger Bridge

D3-Dock moves beyond standard Gaussian Diffusion by solving the pair of forward and backward Stochastic Differential Equations (SDEs) that interpolate between noise and the bound state:

$$d X_t = [f(X_t, t) + g^2(t) \nabla \log \Psi(X_t, t)] dt + g(t) d W_t$$

### Dual-Path Joint Denoising

The core innovation is the joint evolution of continuous geometric space and discrete topological space:

- **Continuous Path (Coordinates):** Estimates the score function $\nabla \log p(x_t | \text{Pocket})$ to refine the $x, y, z$ positions of atoms while adhering to a **Physical Collision Penalty** $V_{\text{collision}}$ based on the protein surface SDF.
- **Discrete Path (Topology):** Utilizes a **D3PM** framework with a structured transition matrix $Q_t \in \mathbb{R}^{K \times K}$ to govern the probability of atom/bond "flipping" between categories (e.g., $C \rightarrow N$).

$$[Q_t]_{ij} = q(z_t = j | z_{t-1} = i)$$

By conditioning the **Discrete Transition Matrix** on the protein surface's local geometry and "chemical texture," we ensure the generated ligand is not just "valid chemistry," but a high-affinity binder.

## 📚 Research Context

This project addresses the bottlenecks identified in current state-of-the-art models:

- **MiDi (2023) & DiffDock (2024):** D3-Dock eliminates the "Physical Hallucination" problem where models generate valid atoms but impossible bond distances by enforcing discrete valency constraints.
- **DiffBindFR (2025):** While DiffBindFR excels at receptor flexibility, D3-Dock provides superior **Topological Integrity** through its hybrid discrete-continuous bridge.
- **SurfDock:** D3-Dock incorporates **Surface Awareness** as a continuous boundary constraint, rather than just a shape-matching heuristic.

**The Bottleneck Solved:** D3-Dock eliminates the "Physical Hallucination" problem where models generate valid atoms but impossible bond distances by enforcing discrete valency constraints in the latent bridge.
