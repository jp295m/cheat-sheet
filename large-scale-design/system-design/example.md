# Example: Backpropagate Computational Graph

## Background

You work at a systematic hedge fund. A method uses deep learning and pathwise forward–backward stochastic differential equations to compute Delta and Gamma of portfolios. A model calibration uses gradient-based optimization routines[^01]. Efficient GPU‑accelerated computation, implemented in C++ and PyTorch, must process thousands of contracts. The segment is **contiguous in execution order**. You want to store the whole graph. Your computation is a **strict unary chain** - every operation takes exactly one input and produces exactly one output. Support append-only growth and dropping old nodes[^02]. Make it easy to mark checkpoints, split segments, and traverse during recomputation.

The bump-and-reprice[^03] approach to calculating Greeks can be computationally expensive. The cost of bumping is $1 + N$ times the cost of a single valuation, where $N$ is the number of model parameters. The adjoint method, by contrast, computes all sensitivities in one sweep with no branching.

## The Tape

The tape must support a method `traverse` that can truncate a sublist from the middle of the tape[^04] for partial recomputation. We denote all of the nodes in the computation graph in the topological ordering - the topological ordering is the ordering where predecessors come before successors, enabling the backward pass for derivative computation.

## Backpropagation[^05]

These adjoint sensitivity methods go under the name of backpropagation in machine learning and **AAD** in finance.

## A Computational Graph

**Model the derivative pricing pipeline as a computational graph** built in **PyTorch** to leverage its autograd engine. The adjoint method steps through time going forward to the payoff; despite the stochastic[^06] fluctuations, the method remains robust.

## The Optimization

Here's the low-level twist: in the forward phase, the autograd tape will remember all the operations it executed, and in the backward phase, it'll replay the operations. This sequence is called the **tape**. Incorporate a core data structure: a sequence of post-maturity cash flow points that moves forward, while the goal of backpropagation is to calculate gradients by tracing from the output back to each parameter. Explain why a simple array is insufficient, and detail how you'd integrate a pointer-based structure into the PyTorch computation graph without breaking gradient[^07] flow. A Bermudan swaption has a schedule of exercise dates. The cash‑flow engine builds a linked list where each node holds a date, a discount factor, and a tensor of exercise value. The list is built dynamically inside `torch.autograd.Function.forward`. Explain how you'd keep this structure on the tape so that the backward pass can correctly accumulate gradients through the list’s tensor fields. If you're operating in a memory-constrained environment[^08], describe how you'd use tape checkpointing to trade off recomputation for memory.

## Your Task

Your main task is to design the in‑memory tape structure that supports pruning and checkpointing. Sketch the data structures, explain how you integrate with PyTorch autograd, and discuss the **memory‑compute tradeoffs**. The broader **system architecture[^09] can be described** in one paragraph.

- Explain how you'd apply the chain rule to propagate the sensitivities backward through the graph until the sensitivities with respect to the inputs are achieved. Give the formula for the adjoint of a quantity on a given node.
- How do you handle non-differentiable payoffs[^10]?

[^01]: sensitivities to hundreds of market parameters
[^02]: sliding window
[^03]: finite differences
[^04]: knocking out a barrier and removing all downstream nodes
[^05]: the AAD layer
[^06]: Black-Scholes, Heston
[^07]: the `.grad_fn` attribute of each torch; tensor is an entry point into this graph
[^08]: GPU/TPU
[^09]: data ingestion, training, serving, monitoring
[^10]: digital options or barrier hits
