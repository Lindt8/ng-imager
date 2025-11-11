"""
phits_usrdef_2_legacy.py

Convert PHITS custom tally output into the event format expected by the
legacy NOVO imaging code.

For now this is just a skeleton: it only parses CLI arguments and sets up
a main() entry point.
"""

import numpy as np 
import pickle 
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PHITS user-defined tally output to NOVO legacy imaging format."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the PHITS user-defined tally output file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Path to the converted output file. "
            "If omitted, a default will be created in the same directory as the input "
            "with '_imaging_records.pickle' appended to the input basename."
        ),
    )

    return parser.parse_args()

def determine_output_path(input_path: Path, output_path: Path | None):
    """
    Determine the final output .pickle path.

    Rules:
    - If output_path is None:
        same directory as input, filename = input basename + '_imaging_records.pickle'
        e.g. usrdef.out -> usrdef.out_imaging_records.pickle
    - If output_path is provided:
        ensure it ends with '.pickle'; if not, append '.pickle'.
    """
    if output_path is None:
        #default_name = input_path.stem + "_imaging_records.pickle"
        default_name = "imaging_data_records.pickle"
        return input_path.parent / default_name

    p = output_path
    if not str(p).endswith(".pickle"):
        # Append .pickle rather than replacing existing extension(s)
        p = p.with_name(p.name + ".pickle")
    return p


def convert_phits_to_legacy(input_path: Path, output_path: Path) -> None:
    """
    Stub for the actual conversion logic.

    Parameters
    ----------
    input_path : Path
        PHITS custom tally output.
    output_path : Path
        Destination file in the legacy imaging format.
    """
    neutron_event_record_type = np.dtype([('type', 'S1'),
                                          ('x1', np.single), ('y1', np.single), ('z1', np.single), ('t1', np.single),
                                          ('dE1', np.single),
                                          ('x2', np.single), ('y2', np.single), ('z2', np.single), ('t2', np.single),
                                          ('det1', np.short), ('det2', np.short),
                                          ('psd1', np.single), ('psd2', np.single),
                                          ('Elong1', np.single), ('Elong2', np.single),
                                          ('protons_only', np.bool_), ('theta_MCtruth', np.single)])
    gamma_event_record_type = np.dtype([('type', 'S1'),
                                        ('x1', np.single), ('y1', np.single), ('z1', np.single), ('dE1', np.single),
                                        ('x2', np.single), ('y2', np.single), ('z2', np.single), ('dE2', np.single),
                                        ('x3', np.single), ('y3', np.single), ('z3', np.single), ('dE3', np.single),
                                        ('t1', np.single), ('t2', np.single), ('t3', np.single),
                                        ('det1', np.short), ('det2', np.short), ('det3', np.short),
                                        ('psd1', np.single), ('psd2', np.single), ('psd3', np.single),
                                        ('Elong1', np.single), ('Elong2', np.single), ('Elong3', np.single),
                                        ('theta1_MCtruth', np.single), ('theta2_MCtruth', np.single)])

    # First pass: count neutron and gamma events to allocate exact-size arrays
    n_neutron_events = 0
    n_gamma_events = 0
    with input_path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            line = line.strip()
            if len(line) == 0: continue  # skip blank lines
            if li < 4: continue  # skip header lines
            line = line[:-1]  # exclude trailing comma
            line_info, _ = line.split(";")
            event_type, iomp, batch, history, phitsno, phitsname = line_info.split()
            if event_type == "ne":
                n_neutron_events += 1
            elif event_type == "ge":
                n_gamma_events += 1
    
    n_im_recs = np.empty((n_neutron_events), dtype=neutron_event_record_type)  # neutron imaging records
    n_im_recs_exp = np.empty((n_neutron_events), dtype=neutron_event_record_type)  # neutron imaging records, but with coordinates at bar centers
    i_nir = 0  # current index of neutron imaging records
    g_im_recs = np.empty((n_gamma_events), dtype=gamma_event_record_type)  # gamma imaging records
    g_im_recs_exp = np.empty((n_gamma_events), dtype=gamma_event_record_type)  # gamma imaging records, but with coordinates at bar centers
    i_gir = 0  # current index of gamma imaging records
    
    with input_path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            line = line.strip()  
            if len(line)==0: continue  # skip blank lines
            if li < 4: continue  # skip header lines
            line = line[:-1]  # exclude trailing comma
            
            line_info, line_content = line.split(';')
            event_type, iomp, batch, history, phitsno, phitsname = line_info.split()
            #hits = line_content.split(',')
            hits = [h for h in line_content.split(',') if h.strip()]
            if event_type == 'ne':
                hits_iter = sorted(
                    hits,
                    key=lambda h: float(h.split()[-1])  # t is the last field
                )
            else:
                hits_iter = hits
            
            for ih, hit in enumerate(hits_iter, start=1):
                reg, edep, x, y, z, t = hit.split()
                if event_type == 'ne':  # neutron event
                    if ih > 2: continue  # code not prepared for >2x neutron coincs
                    if ih==1:
                        n_im_recs['type'][i_nir] = 'n'
                        n_im_recs[f'dE{ih}'][i_nir] = edep
                        n_im_recs['protons_only'][i_nir] = True  # we don't actually know this for short version of usrdef.out
                        n_im_recs['theta_MCtruth'][i_nir] = None  # not knowable since we don't have neutron origin location
                    n_im_recs[f'x{ih}'][i_nir] = x
                    n_im_recs[f'y{ih}'][i_nir] = y
                    n_im_recs[f'z{ih}'][i_nir] = z
                    n_im_recs[f't{ih}'][i_nir] = t
                    n_im_recs[f'det{ih}'][i_nir] = reg
                    n_im_recs[f'Elong{ih}'][i_nir] = edep
                    n_im_recs[f'psd{ih}'][i_nir] = None
                elif event_type == 'ge':
                    if ih > 3: continue  # code not prepared for >3x gamma coincs
                    if ih==1:
                        g_im_recs['type'][i_gir] = 'g'
                        g_im_recs['theta1_MCtruth'][i_gir] = None  # not knowable since we don't have neutron origin location
                        g_im_recs['theta2_MCtruth'][i_gir] = None  # not knowable since we don't have neutron origin location
                    g_im_recs[f'x{ih}'][i_gir] = x
                    g_im_recs[f'y{ih}'][i_gir] = y
                    g_im_recs[f'z{ih}'][i_gir] = z
                    g_im_recs[f't{ih}'][i_gir] = t
                    g_im_recs[f'dE{ih}'][i_gir] = edep
                    g_im_recs[f'det{ih}'][i_gir] = reg
                    g_im_recs[f'Elong{ih}'][i_gir] = edep
                    g_im_recs[f'psd{ih}'][i_gir] = None
            if event_type == 'ne':
                n_im_recs_exp[i_nir] = n_im_recs[i_nir]
                i_nir += 1
            elif event_type == 'ge':
                g_im_recs_exp[i_gir] = g_im_recs[i_gir]
                i_gir += 1
        
    with open(output_path, 'wb') as handle:
        to_be_pickled = {'neutron_records': n_im_recs, 'gamma_records': g_im_recs,
                         'neutron_records_exp': n_im_recs_exp, 'gamma_records_exp': g_im_recs_exp,
                         'sim_base_folder_name': input_path}
        pickle.dump(to_be_pickled, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickle file written:', output_path, '\n')


def main() -> None:
    args = parse_args()

    # Basic existence checks (you can relax these later if you want to read from stdin)
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Resolve the final output path (apply defaults and .pickle extension logic)
    output_path = determine_output_path(args.input, args.output)

    # Ensure output directory exists
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_phits_to_legacy(args.input, output_path)


if __name__ == "__main__":
    main()






