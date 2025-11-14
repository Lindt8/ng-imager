from ngimager.config.schemas import Config
from ngimager.geometry.plane import Plane
from ngimager.io.lm_store import write_init, write_summed, read_summed
import numpy as np, tempfile, os


def test_hdf5_write_read():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    tmp.close()

    class Dummy:
        pass

    cfg = Dummy()
    cfg_path = __file__  # any text file

    # Plane object is built directly here; cfg.plane is not used in this test.
    pl = Plane.from_cfg([0, 0, 0], [0, 0, 1], -1, 1, 0.5, -1, 1, 0.5)

    cfg_obj = Config(
        run={
            # new RunCfg shape
            "source_type": "proton_center",
            "workers": "auto",
            "chunk_cones": "auto",
            "jit": False,
            "progress": False,
            "diagnostics_level": 0,  # keep test quiet
            # neutrons/gammas/fast/list all take defaults
        },
        io={
            # new IOCfg shape
            "input_path": "in",
            "input_format": "phits",
            "output_path": "out",
        },
        plane={
            "origin": [0, 0, 0],
            "normal": [0, 0, 1],
            "u_min": -1,
            "u_max": 1,
            "du": 0.5,
            "v_min": -1,
            "v_max": 1,
            "dv": 0.5,
        },
        filters={
            "min_light": 0,
            "max_light": 1e9,
            "tof_window_ns": [0, 1e6],
            "bars_include": [],
            "materials_include": [],
        },
        energy={
            "strategy": "ELUT",
            "fixed_En_MeV": 14.1,
            "lut_paths": {},
        },
        prior={
            "type": "point",
            "point": [0, 0, 0],
            "strength": 1.0,
        },
        uncertainty={
            "enabled": False,
            "smearing": "thicken",
            "sigma_doi_cm": 0.35,
            "sigma_transverse_cm": 0.346,
            "sigma_time_ns": 0.5,
            "use_lut_bands": False,
        },
        # detectors, vis, pipeline all use defaults
    )

    f = write_init(tmp.name, cfg_path, cfg_obj, pl)

    img = np.zeros((pl.nv, pl.nu), dtype="u4")
    img[0, 0] = 1
    write_summed(f, "n", img)
    f.close()

    arr = read_summed(tmp.name, "n")
    os.unlink(tmp.name)
    assert arr.shape == img.shape and arr[0, 0] == 1
