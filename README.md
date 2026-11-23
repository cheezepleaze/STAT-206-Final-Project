# JackknightPlus.jl

A Julia package for distribution-free predictive inference and unvertainty quantification. It implements the Jackknife+ framework ([Barber et al., 2021](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.pdf)) with OLS leverage-score optimization.

## Authors

* Saulo Bosquez
* Adrian Muli

## Overview

`JackknifePlus.jl` provides non-asymptotic prediction intervals with theoretical guaranteed coverage floors ($1 - 2\alpha$) without requiring distribution or parametric assumptions. 

## Installation

To install and activate the package in your local Julia environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()