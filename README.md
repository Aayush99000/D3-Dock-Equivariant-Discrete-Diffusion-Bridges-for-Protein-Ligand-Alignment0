# D3-Dock: Equivariant Discrete Diffusion for Chemically-Consistent Molecular Docking

D3-Dock is a next-generation generative framework for protein-ligand docking. While previous models like MiDi (2023) often suffer from **structural dissonance**—a mismatch between discrete chemical bonds and continuous 3D coordinates—D3-Dock uses a **Discrete Denoising Diffusion Probabilistic Model (D3PM)** to "guardrail" the folding process.

## Key Innovations

- **Discrete-Led Refinement:** Uses a transition matrix $Q_t$ to evolve atom and bond types discretely, ensuring $100\%$ chemical valency throughout the diffusion process.
- **$E(3)$-Equivariant Architecture:** The underlying Graph Neural Network is rotationally and translationally invariant, respecting the fundamental physics of 3D space.
- **Schrödinger Bridge Coupling:** Instead of unconstrained generation, D3-Dock finds the optimal transport path between random noise and a target protein pocket.
- **Parallel Generation:** Bypasses the $O(N)$ bottleneck of autoregressive models, refining the entire molecular topology in $O(1)$ parallel steps.

## Technical Architecture

D3-Dock solves the joint distribution of topology and geometry:

$$q(z_t | z_0, \text{Pocket}) = f(z_t^{\text{continuous}}, z_t^{\text{discrete}}, \text{Pocket Condition})$$

By conditioning the **Discrete Transition Matrix** on the protein pocket's local geometry, we ensure that the generated ligand is not just "valid chemistry," but a high-affinity binder for that specific pocket.

## 📚 Research Context

This project builds upon and addresses the bottlenecks found in:

- **MiDi:** Mixed Discrete-Continuous Diffusion Models for Graph Generation (Vignac et al., 2023)
- **DiffDock:** Diffusion Steps, Twists, and Turns for Molecular Docking (Corso et al., 2024)

**The Bottleneck Solved:** D3-Dock eliminates the "Physical Hallucination" problem where models generate valid atoms but impossible bond distances by enforcing discrete valency constraints in the latent bridge.
