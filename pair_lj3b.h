/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj3b,PairLJ3B);
// clang-format on
#else

#ifndef LMP_PAIR_LJ3B_H
#define LMP_PAIR_LJ3B_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJ3B : public Pair {
 public:
  PairLJ3B(class LAMMPS *);
  ~PairLJ3B() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

  static constexpr int NPARAMS_PER_LINE = 9;

  struct Param {
    double epsilon, sigma;
    double cutoff, lambda;
    double powern;
    int flagoffset;    //offset flag
    double cut, cutsq;
    double c1, c2, c3, c4;
    int ielement, jelement, kelement;
  };

 protected:
  double cutmax;      // max cutoff for all elements
  Param *params;      // parameter set for an I-J-K interaction
  int maxshort;       // size of short neighbor list array
  int *neighshort;    // short neighbor list array

  void settings(int, char **) override;
  virtual void allocate();
  virtual void read_file(char *);
  virtual void setup_params();
  void twobody(Param *, double, double &, int, double &);
  virtual void threebody(Param *, Param *, Param *, double, double, double *, double *, double *,
                         double *, int, double &);
};

}    // namespace LAMMPS_NS

#endif
#endif
