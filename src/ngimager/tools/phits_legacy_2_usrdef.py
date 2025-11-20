"""
phits_legacy_2_usrdef.py

Convert a NOVO legacy imaging pickle file (containing neutron_records and
gamma_records) into a PHITS usrdef-short style text file that can be parsed
directly by the modern ng-imager PHITS adapter
(parse_phits_usrdef_short / from_phits_usrdef).

This script is the structural inverse of phits_usrdef_2_legacy.py.

USAGE
-----

Basic conversion:

    $ python phits_legacy_2_usrdef.py legacy_imaging_records.pickle

This produces:

    legacy_imaging_records_usrdef.out

in the same directory as the input file.

Specify an explicit output file:

    $ python phits_legacy_2_usrdef.py legacy_imaging_records.pickle -o my_usrdef.out


INPUT FORMAT
------------

The input must be a pickle file produced by the legacy imaging pipeline or by
phits_usrdef_2_legacy.py. It must contain (at minimum) the keys:

    'neutron_records'   -- numpy structured array of 2-hit neutron events
    'gamma_records'     -- numpy structured array of 3-hit gamma events

The dtype layouts follow the legacy NOVO event record definitions.

OUTPUT FORMAT
-------------

The output is a plain-text file in PHITS usrdef-short format, where each line
has the structure:

    event_type  iomp  batch  hist  no  name  ;  reg Edep x y z t  ,  reg Edep x y z t  , ...

For example:

    ne 0 0 0 12 0 ;  200 0.45 -3.5 16.2 0.4 4.70  ,  210 0.31 -3.3 16.8 0.6 4.78  ,

    ge 0 0 0 7  0 ;  101 0.20  2.1 -3.0 0.9 7.50  ,  105 0.12 2.8 -3.3 1.1 7.56 ,
                         110 0.07  3.4 -4.1 1.4 7.63  ,

These lines are fully compatible with:

    parse_phits_usrdef_short(path)
    from_phits_usrdef(path)
    PHITSAdapter(...).iter_raw_events(path)

NOTES
-----

• PHITS bookkeeping fields (iomp, batch, hist, name) are assigned default
  placeholder values; feel free to extend the script to generate structured
  metadata.

• Legacy dE/Elong values are written as Edep(MeV); the ng-imager parser
  treats energy fields transparently.

• This converter is intended for analysis, debugging, and round-trip testing
  between the legacy pipeline and the modern ng-imager framework.

"""


import pickle
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert NOVO legacy imaging pickle → PHITS usrdef-short text."
    )

    ap.add_argument(
        "input",
        type=Path,
        help="Legacy imaging pickle file (neutron_records / gamma_records).",
    )

    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination usrdef-style text file. Defaults to input.stem + '_usrdef.out'",
    )

    return ap.parse_args()


def determine_output_path(input_path: Path, output: Path | None) -> Path:
    if output is None:
        return input_path.parent / f"{input_path.stem}_usrdef.out"
    return output


def format_hit(reg: int, edep: float, x: float, y: float, z: float, t: float) -> str:
    """
    Return one PHITS hit group as:
        reg  Edep(MeV)  x(cm)  y(cm)  z(cm)  t(ns)
    (Space-separated, no commas — the adapter ignores commas and semicolons anyway.)
    """
    return f"{reg:d} {edep:.6g} {x:.6g} {y:.6g} {z:.6g} {t:.6g}"


def convert_legacy_to_usrdef(input_path: Path, output_path: Path) -> None:
    """
    Convert legacy pickle → usrdef-short compatible text.

    Output rows follow the format:

        event_type  iomp  batch  hist  no  name  ;  hit1  ,  hit2  , ...

    where:
      • event_type = 'ne' or 'ge'
      • hit = 6 floating fields: reg Edep x_cm y_cm z_cm t_ns
    """
    
    header = "!Required final counter 1 value =       2 ; Required final counter 2 value =       3\n" +\
        "!       #iomp    #batch  #history       #no     #name        #reg  EdepA(MeV)      xA(cm)      yA(cm)      zA(cm)      tA(ns)        #reg  EdepB(MeV)      xB(cm)      yB(cm)      zB(cm)      tB(ns)        #reg  EdepC(MeV)      xC(cm)      yC(cm)      zC(cm)      tC(ns)\n" +\
        "!ncol   Z   N jcl kcl nclsts\n" +\
        "!In/Out kf-code     E(MeV)      weight \n"

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    n_rec = data.get("neutron_records", [])
    g_rec = data.get("gamma_records", [])

    # PHITS bookkeeping placeholders
    iomp = 0
    batch = 0
    hist = 0
    name = 0

    lines = []

    # -------------------------
    # Neutrons → "ne"
    # -------------------------
    for idx, ev in enumerate(n_rec):
        # Legacy dtype schema ensures x1,y1,z1,t1,dE1,det1,Elong1 exist
        # Combine dE1 and Elong1 possibility: use Elong1 as Edep(MeV) proxy
        hits = []

        # first scatter
        hits.append(format_hit(
            int(ev["det1"]),
            float(ev["Elong1"]),   # PHITS expects MeV; Elong is MeVee but this is acceptable for round-trip
            float(ev["x1"]),
            float(ev["y1"]),
            float(ev["z1"]),
            float(ev["t1"]),
        ))

        # second scatter
        hits.append(format_hit(
            int(ev["det2"]),
            float(ev["Elong2"]),
            float(ev["x2"]),
            float(ev["y2"]),
            float(ev["z2"]),
            float(ev["t2"]),
        ))

        line = f"ne {iomp} {batch} {hist} {idx} {name} ; " + " , ".join(hits) + " ,"
        lines.append(line)

    # -------------------------
    # Gammas → "ge"
    # -------------------------
    for idx, ev in enumerate(g_rec):
        hits = []
        
        try:
            d1 = int(ev["det1"])
        except:
            d1 = 1
        try:
            d2 = int(ev["det2"])
        except:
            d2 = 2
        try:
            d3 = int(ev["det3"])
        except:
            d3 = 3
            
        hits.append(format_hit(
            d1, float(ev["dE1"]),
            float(ev["x1"]), float(ev["y1"]), float(ev["z1"]),
            float(ev["t1"]),
        ))
        hits.append(format_hit(
            d2, float(ev["dE2"]),
            float(ev["x2"]), float(ev["y2"]), float(ev["z2"]),
            float(ev["t2"]),
        ))
        hits.append(format_hit(
            d3, float(ev["dE3"]),
            float(ev["x3"]), float(ev["y3"]), float(ev["z3"]),
            float(ev["t3"]),
        ))

        line = f"ge {iomp} {batch} {hist} {idx} {name} ; " + " , ".join(hits) + " ,"
        lines.append(line)

    # -------------------------
    # Write file
    # -------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for L in lines:
            f.write(L + "\n\n")

    print(f"usrdef-style file written: {output_path}")


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input pickle not found: {args.input}")

    out = determine_output_path(args.input, args.output)
    convert_legacy_to_usrdef(args.input, out)


if __name__ == "__main__":
    main()
