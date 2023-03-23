// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Rodrigo
------------------------------------------------------------------------- */

#include "pair_fs3b.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "potential_file_reader.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define DELTA 4

/* ---------------------------------------------------------------------- */

PairFS3B::PairFS3B(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  params = nullptr;

  maxshort = 10;
  neighshort = nullptr;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairFS3B::~PairFS3B()
{
  if (copymode) return;

  memory->destroy(params);
  memory->destroy(elem3param);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
  }
}

/* ---------------------------------------------------------------------- */

void PairFS3B::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum,jnumm1;
  int itype,jtype,ktype,ijparam,ikparam,ijkparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rsq1,rsq2;
  double delr1[3],delr2[3],fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rmin,vnorm;  //new

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      jtype = map[type[j]];
      ijparam = elem3param[itype][jtype][jtype];
      if (rsq >= params[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      twobody(&params[ijparam],rsq,fpair,eflag,evdwl);

      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;
      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,0.0,fpair,delx,dely,delz);
    }

    jnumm1 = numshort - 1;

    for (jj = 0; jj < jnumm1; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      ijparam = elem3param[itype][jtype][jtype];
      if (params[ijparam].flagoffset == 1 ) {
         continue;
      }
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      double fjxtmp,fjytmp,fjztmp;
      fjxtmp = fjytmp = fjztmp = 0.0;

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        ikparam = elem3param[itype][ktype][ktype];
        ijkparam = elem3param[itype][jtype][ktype];
        // skip for lambda equals to zero
        if (params[ijkparam].lambda <= 0.00001 ) {
           continue;
        }
        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        threebody(&params[ijparam],&params[ikparam],&params[ijkparam],
                  rsq1,rsq2,delr1,delr2,fj,fk,eflag,evdwl);

        fxtmp -= fj[0] + fk[0];
        fytmp -= fj[1] + fk[1];
        fztmp -= fj[2] + fk[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

        if (evflag) ev_tally3(i,j,k,evdwl,0.0,fj,fk,delr1,delr2);
      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairFS3B::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairFS3B::settings(int narg, char **/*arg*/)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairFS3B::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg-3,arg+3);

  // read potential file and initialize potential parameters

  read_file(arg[2]);
  setup_params();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairFS3B::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style FS3B requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style FS3B requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairFS3B::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairFS3B::read_file(char *file)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, file, "fs3b", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {
      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();
        std::string kname = values.next_string();

        // ielement,jelement,kelement = 1st args
        // if all 3 args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement, kelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;
        for (kelement = 0; kelement < nelements; kelement++)
          if (kname == elements[kelement]) break;
        if (kelement == nelements) continue;

        // load up parameter settings and error check their values

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
                                              "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA*sizeof(Param));
        }

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].kelement = kelement;
        params[nparams].epsilon  = values.next_double();
        params[nparams].sigma    = values.next_double();
        params[nparams].littlea  = values.next_double();
        params[nparams].lambda   = values.next_double();
        params[nparams].bigb     = values.next_double();
        params[nparams].powerp   = values.next_double();
        params[nparams].powerq   = values.next_double();
        params[nparams].tol      = values.next_double();
	params[nparams].flagoffset  = values.next_int();
      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      if (unit_convert) {
        params[nparams].epsilon *= conversion_factor;
      }

      if (params[nparams].epsilon < 0.0 || params[nparams].sigma < 0.0 ||
          params[nparams].littlea < 0.0 || params[nparams].lambda < 0.0 ||
          params[nparams].bigb < 0.0 || params[nparams].powerp < 0.0 ||
          params[nparams].powerq < 0.0 || params[nparams].tol < 0.0 ||
          !(params[nparams].flagoffset == 0 ||
          params[nparams].flagoffset == 1 ) )
        error->one(FLERR,"Illegal FS3B parameter");

      if (params[nparams].flagoffset == 1 && params[nparams].lambda > 0.0 )
        error->one(FLERR,"FS3B offset and lambda incompatible, both can't be ON");
      if (params[nparams].flagoffset == 1 && params[nparams].epsilon == 0.0 )
        error->one(FLERR,"FS3B offset and epsilon incompatible, epsilon can't be 0");
      if (params[nparams].flagoffset == 1 && params[nparams].sigma == 0.0 )
        error->one(FLERR,"FS3B offset and sigma incompatible, sigma can't be 0");

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params,maxparam*sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam*sizeof(Param), MPI_BYTE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairFS3B::setup_params()
{
  int i,j,k,m,n;
  double rtmp;

  // set elem3param for all triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem3param);
  memory->create(elem3param,nelements,nelements,nelements,"pair:elem3param");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nparams; m++) {
          if (i == params[m].ielement && j == params[m].jelement &&
              k == params[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem3param[i][j][k] = n;
      }


  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  for (m = 0; m < nparams; m++) {
    params[m].cut = params[m].sigma*params[m].littlea;

    params[m].c1 = params[m].powerp*params[m].bigb *
                   pow(params[m].sigma,params[m].powerp);
    params[m].c2 = params[m].powerq *
                   pow(params[m].sigma,params[m].powerq);
    params[m].c3 = params[m].bigb *
                   pow(params[m].sigma,params[m].powerp+1.0);
    params[m].c4 = pow(params[m].sigma,params[m].powerq+1.0);
    params[m].c5 = params[m].bigb *
                   pow(params[m].sigma,params[m].powerp);
    params[m].c6 = pow(params[m].sigma,params[m].powerq);


    //add here the rmin and vnorm
    //rmin:
    double r,rsq,rinv,rp,rq,rainv,rainvsq,expsrainv,function;
    double ra,rb;
    double delta;
    int n,nmax;
    //initial conditions
    delta = 0.001;
    nmax = 200;
    ra = params[m].sigma*0.1;  //function positive when evaluated
    rb = params[m].cut - delta;  //function negative when evaluated
    //bisection method  
    n = 0;
    if ( params[m].epsilon > 0.0 && params[m].sigma > 0.0 ){
      do {
              r = (ra + rb) / 2.0;
          //function evaluation
              rsq = r*r;
          rinv = 1.0/r;
              rp = pow(r,-params[m].powerp);
              rq = pow(r,-params[m].powerq);
              rainv = 1.0 / (r - params[m].cut);
              rainvsq = rainv*rainv*r;
              expsrainv = exp(params[m].sigma * rainv);
              //the twobody fforce but the last term (1/r*r -> 1/r) 
              function = (params[m].c1*rp - params[m].c2*rq +
                         (params[m].c3*rp - params[m].c4*rq) * rainvsq) *
                         expsrainv * rinv;
          if ( (function*function < delta*delta) ||
             (rb - ra < delta*delta) ) {
                      //fmt::print(screen,"rmin is: {} for m:{} \n",r,m); //test functions
                      params[m].rmin = r;
                      break;
              } else {
              if (function>0){
                  ra = r;
              } else {
                  rb = r;
              }
              }
          n = n + 1;
      }
      while (n<nmax);
          if ( n == nmax )
      error->all(FLERR,"In setup_params(): rmin not found, more than nmax loops performed");
      //vnorm part:
      //the (negative) twobody potential energy (without the epsilon) evaluated in rmin
      params[m].vnorm = -(params[m].c5*rp - params[m].c6*rq) *
                        expsrainv;
    } else {
          params[m].rmin = 0.0;
          params[m].vnorm = 1.0;
        }


    rtmp = params[m].cut;
    if (params[m].tol > 0.0) {
      if (params[m].tol > 0.01) params[m].tol = 0.01;
      rtmp = rtmp +
             params[m].sigma / log(params[m].tol);
    }
    //cut selection when offset ON
    if (params[m].flagoffset == 1) {
            // take out : params[m].cut = params[m].rmin; !!!!!!!!!!!!
            // interferes in two(three)body when cut is called again
                rtmp = params[m].rmin;
    }
    params[m].cutsq = rtmp * rtmp;

    //print info 
    //fmt::print(screen,"m: {}, rmin: {}, |vnorm(rmin)|: {}, |v(rmin)|: {}, flag: {}\n",
    //m,params[m].rmin,params[m].vnorm,params[m].epsilon,params[m].flagoffset); 

  }
  //fmt::print(screen,"\n");


  // set cutmax to max of all params

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = sqrt(params[m].cutsq);
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

/* ---------------------------------------------------------------------- */

void PairFS3B::twobody(Param *param, double rsq, double &fforce,
                     int eflag, double &eng)
{
  double r,rinvsq,rp,rq,rainv,rainvsq,expsrainv;

  r = sqrt(rsq);
  rinvsq = 1.0/rsq;
  rp = pow(r,-param->powerp);
  rq = pow(r,-param->powerq);
  rainv = 1.0 / (r - param->cut);
  rainvsq = rainv*rainv*r;
  expsrainv = exp(param->sigma * rainv);
  fforce = param->epsilon * (param->c1*rp - param->c2*rq +
           (param->c3*rp -param->c4*rq) * rainvsq) * expsrainv *
           rinvsq / param->vnorm;
  if (eflag) eng = ( param->epsilon * (param->c5*rp - param->c6*rq) *
                   expsrainv / param->vnorm ) +
                   param->epsilon * param->flagoffset; // offset/shift energy modification
}

/* ---------------------------------------------------------------------- */

void PairFS3B::threebody(Param *paramij, Param *paramik, Param *paramijk,
                       double rsq1, double rsq2,
                       double *delr1, double *delr2,
                       double *fj, double *fk, int eflag, double &eng)
{
  double r1,rinvsq1,rainv1;
  double r2,rinvsq2,rainv2;
  double rp1,rq1,rainvsq1,expsrainv1,fforce1,eng1;
  double rp2,rq2,rainvsq2,expsrainv2,fforce2,eng2;

  r1 = sqrt(rsq1);
  rinvsq1 = 1.0/rsq1;
  rp1 = pow(r1,-paramij->powerp);
  rq1 = pow(r1,-paramij->powerq);
  rainv1 = 1.0/(r1 - paramij->cut);
  rainvsq1 = rainv1*rainv1*r1;
  expsrainv1 = exp(paramij->sigma * rainv1);
  // twobody without epsilon (normalized) 
  fforce1 = (paramij->c1*rp1 - paramij->c2*rq1 +
            (paramij->c3*rp1 -paramij->c4*rq1) * rainvsq1) *
            expsrainv1 * rinvsq1 / paramij->vnorm;
  // twobody without epsilon (normalized)
  eng1 = (paramij->c5*rp1 - paramij->c6*rq1) * expsrainv1 /
         paramij->vnorm;


  r2 = sqrt(rsq2);
  rinvsq2 = 1.0/rsq2;
  rp2 = pow(r2,-paramik->powerp);
  rq2 = pow(r2,-paramik->powerq);
  rainv2 = 1.0/(r2 - paramik->cut);
  rainvsq2 = rainv2*rainv2*r2;
  expsrainv2 = exp(paramik->sigma * rainv2);
  // twobody without epsilon (normalized)
  fforce2 = (paramik->c1*rp2 - paramik->c2*rq2 +
            (paramik->c3*rp2 -paramik->c4*rq2) * rainvsq2) *
            expsrainv2 * rinvsq2 / paramik->vnorm;
  // twobody without epsilon (normalized)
  eng2 = (paramik->c5*rp2 - paramik->c6*rq2) * expsrainv2 /
         paramik->vnorm;


  //new: 
  //conditional to see if r is below or above rmin
  // r is below rmin if fforce# is greater than zero >0 
  //set the force factor and energy accordingly
  // can implement differently now that rmin is included
  //ij part
  if (fforce1 > 0.0) {  //new
        fforce1 = 0.0;
        eng1 = 1;
  } else {
        fforce1 = -fforce1;
        eng1 = -eng1;
  }
  //ik part
  if (fforce2 > 0.0) {  //new
        fforce2 = 0.0;
        eng2 = 1;
  } else {
        fforce2 = -fforce2;
        eng2 = -eng2;
  }

  fj[0] = paramijk->lambda*paramijk->epsilon*delr1[0]*fforce1*eng2;
  fj[1] = paramijk->lambda*paramijk->epsilon*delr1[1]*fforce1*eng2;
  fj[2] = paramijk->lambda*paramijk->epsilon*delr1[2]*fforce1*eng2;

  fk[0] = paramijk->lambda*paramijk->epsilon*delr2[0]*fforce2*eng1;
  fk[1] = paramijk->lambda*paramijk->epsilon*delr2[1]*fforce2*eng1;
  fk[2] = paramijk->lambda*paramijk->epsilon*delr2[2]*fforce2*eng1;


  if (eflag) eng = paramijk->lambda*paramijk->epsilon*eng1*eng2;


}

/* ---------------------------------------------------------------------- */
