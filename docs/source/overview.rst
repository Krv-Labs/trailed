.. _overview:

========
Overview
========

TRAILED provides **differentiable Euler Characteristic Transform (ECT)** computation with first-class support for scientific Python workflows. It bridges topological data analysis with modern machine learning.

What is ECT?
------------

The **Euler Characteristic Transform** is a topological descriptor that captures shape information through directional filtrations:

1. **Direction sampling**: Choose directions on the unit sphere
2. **Filtration**: Sweep a hyperplane along each direction
3. **Euler characteristic**: Count connected components, holes, and voids at each level
4. **Vectorization**: Concatenate all curves into a fixed-length descriptor

.. mermaid::

   graph LR
      subgraph "Input"
         A["Point cloud"]
      end

      subgraph "Direction Sampling"
         B1["θ₁"]
         B2["θ₂"]
         B3["θₙ"]
      end

      subgraph "Filtration"
         C1["EC curve 1"]
         C2["EC curve 2"]
         C3["EC curve n"]
      end

      subgraph "Output"
         D["Descriptor vector"]
      end

      A --> B1
      A --> B2
      A --> B3
      B1 --> C1
      B2 --> C2
      B3 --> C3
      C1 --> D
      C2 --> D
      C3 --> D

      style A fill:#f9f9f9,stroke:#999
      style D fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

Why ECT?
--------

ECT descriptors have powerful properties:

**Completeness**
   ECT can distinguish between almost all shapes—it's injective on a dense subset of shapes.

**Stability**
   Small perturbations to input shapes produce small changes in descriptors.

**Differentiability**
   TRAILED's implementation supports gradients for end-to-end learning.

**Interpretability**
   Each direction captures different geometric features of the input.

Architecture
------------

TRAILED has a layered design:

.. mermaid::

   graph TB
      subgraph "Core (dect)"
         A["Direction sampling"]
         B["Filtration computation"]
         C["EC curve calculation"]
      end

      subgraph "Adapters"
         D["NumPy interface"]
         E["pandas interface"]
      end

      subgraph "Plugins (optional)"
         F["sklearn transformer"]
         G["PyTorch layer"]
      end

      A --> B
      B --> C
      C --> D
      C --> E
      D --> F
      D --> G

      style A fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style C fill:#D9EDF7,stroke:#31708F,stroke-width:2px

Key Components
--------------

**Direction Sampling**

TRAILED supports multiple sampling strategies:

- **Uniform**: Random directions on the sphere
- **Stratified**: Evenly distributed directions
- **Custom**: User-defined direction sets

**Resolution Control**

The ``resolution`` parameter controls filtration granularity:

- Higher resolution = more detail, larger descriptors
- Lower resolution = faster computation, coarser features

**Framework Integration**

=============== =====================================================
Framework        Installation
=============== =====================================================
NumPy/pandas     ``pip install trailed`` (included)
sklearn          ``pip install trailed[sklearn]``
PyTorch          ``pip install trailed[torch]``
All              ``pip install trailed[all]``
=============== =====================================================

Use Cases
---------

**Shape Classification**
   Train classifiers on ECT descriptors to distinguish shape categories.

**Shape Retrieval**
   Find similar shapes in a database using ECT distance.

**Generative Models**
   Use differentiable ECT as a loss function for shape generation.

**Data Imputation**
   Score imputation candidates by their topological properties (see :doc:`Phil <../../../phil/docs/source/index>`).

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Compute your first descriptor
- :ref:`User Guide <user_guide>` - Detailed configuration
- :ref:`Integrations <integrations>` - sklearn and PyTorch adapters
