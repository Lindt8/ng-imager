# Imaging PHITS-produced data

The imaging dataset here was produced by running the PHITS input file `novo-example.inp` through PHITS that had been recompiled with a custom user-defined tally ([T-Userdefined]) whose source and documentation can be found at: https://github.com/Lindt8/T-Userdefined/tree/main/multi-coincidence_ng

The example's geometry and source is pictured below (`geometry.png`); it consists of a 150 MeV proton beam incident on a water phantom, with 5 detector element cells placed adjacent to the target.  

![](geometry.png)


The produced `usrdef.out` file from this tally contains two-fold neutron coincident and three-fold gamma-ray coincident event data.  This file is then processed by `ng-imager`, and the respective neutrons and gamma rays are imaged.

