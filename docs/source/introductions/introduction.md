# Introduction
HYPO is a physical optics package for modeling and analyzing electromagnetic propagation in optical systems. It is developed to support the study of beam formation and polarization properties in millimeter-wave and sub-millimeter instruments, where diffraction, phase evolution, dielectric-interface effects, and polarization-dependent responses must be treated consistently.

- The framework is intended for three classes of optical systems:

- Purely refractive optics, including dielectric elements and interfaces with anti-reflection (AR) coatings;

- Purely reflective systems, such as reflector antennas and telescope mirrors, planned for future releases;

- Hybrid refractive-reflective systems, in which dielectric and reflective elements jointly determine the final beam, also planned for future releases.

The current version of HYPO focuses on refractive optical systems. A key feature of the package is that it accounts for the electromagnetic response of dielectric interfaces, including the effects of AR coatings. This makes it possible to analyze not only beam intensity and phase distributions, but also polarization-related effects such as polarization-dependent transmission, reflection, and beam distortions introduced by refractive components.

The main purpose of HYPO is to provide a unified framework for studying:
- beam propagation through optical systems,
- near-field and far-field beam patterns,
- amplitude and phase evolution,
- polarization properties and instrumental polarization effects,
- the impact of coated optical surfaces on system performance.

By developing refractive, reflective, and hybrid capabilities within a common physical optics framework, HYPO aims to provide a consistent tool for end-to-end electromagnetic analysis across a wide range of optical designs. In the current release, this capability is implemented for refractive optics, while support for reflective and mixed systems is planned for later versions.

The documentation introduces the scope of the current release, installation steps, core concepts, and example workflows.