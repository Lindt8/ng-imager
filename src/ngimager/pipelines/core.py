'''
The CLI is implemented via Typer, so the script behaves like a simple one-argument command:

    - argument = path to TOML config file

The pipeline will:

    - Load the TOML config
    - Detect the adapter (PHITS, ROOT, HDF5 restart)
    - Shape/validate hits → events
    - Build cones (now neutron + gamma depending on run.neutrons, run.gammas)
    - Run SBP imaging
    - Write unified HDF5 output

You can always show help via:

    - `python -m ngimager.pipelines.core --help`

Example run commands from project root:

python -m ngimager.pipelines.core path/to/config.toml

python -m ngimager.pipelines.core examples/configs/phits_usrdef_simple.toml

python -m ngimager.pipelines.core .\examples\configs\phits_usrdef_simple.toml


'''
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Literal, Optional, Dict
import typer

import numpy as np

from ngimager.config.load import load_config
from ngimager.config.schemas import Config, RunCfg
from ngimager.geometry.plane import Plane
from ngimager.geometry.transforms import apply_rigid_transform, is_identity_transform
from ngimager.io.adapters import make_adapter
from ngimager.io.lm_store import (
    write_init,
    write_summed,
    write_projections,
    write_cones,
    write_lm_indices,
    write_events_hits,
    write_counters,
    write_event_cone_survival,
    write_root_novo_meta,
)
from ngimager.io.lut import build_lut_registry
from ngimager.imaging.sbp import reconstruct_sbp, Cone, cone_to_indices
from ngimager.physics.cones import build_cone_from_neutron, build_cone_from_gamma
from ngimager.physics.events import NeutronEvent, GammaEvent, Event
from ngimager.physics.energy_strategies import make_energy_strategy
from ngimager.physics.priors import make_prior, Prior
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
from ngimager.filters.to_typed_events import shaped_to_typed_events
from ngimager.filters.hit_filters import apply_hit_filters, is_reconstructable
from ngimager.filters.cone_filters import passes_delta_theta_cut
from ngimager.filters.event_filters import apply_event_filters
from ngimager.vis.hdf import render_summed_images

def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    """
    Increment a counter by delta, creating it if needed.
    """
    counters[key] = counters.get(key, 0) + delta

def _apply_fast_overrides(cfg: Config, diag_level: int = 1) -> None:
    """
    Apply [fast] overrides when [run].fast is true.

    This mutates cfg in-place:
      - adjusts event-level light thresholds,
      - caps the number of cones,
      - coarsens the imaging plane grid.
    """
    if not getattr(cfg.run, "fast", False):
        return

    fast_cfg = getattr(cfg, "fast", None)
    if fast_cfg is None:
        return

    if diag_level >= 1:
        print("[run] fast mode enabled; applying [fast] overrides")

    # ---- Event-level light thresholds ---------------------------------
    events_cfg = getattr(cfg.filters, "events", None)
    
    if events_cfg is not None:
        events_n_cfg = getattr(cfg.filters, "neutron", None)
        events_g_cfg = getattr(cfg.filters, "gamma", None)
        if fast_cfg.min_L1_MeVee is not None and events_n_cfg is not None:
            events_n_cfg.min_L1_MeVee = fast_cfg.min_L1_MeVee
        if fast_cfg.min_L2_MeVee is not None and events_n_cfg is not None:
            events_n_cfg.min_L2_MeVee = fast_cfg.min_L2_MeVee
        if fast_cfg.min_L_any_MeVee is not None and events_g_cfg is not None:
            events_g_cfg.min_L_any_MeVee = fast_cfg.min_L_any_MeVee

    # ---- Max cones cap -----------------------------------------------
    if fast_cfg.max_cones is not None:
        cfg.run.max_cones = fast_cfg.max_cones

    # ---- Imaging plane downsampling -----------------------------------
    if fast_cfg.plane_downsample is not None and fast_cfg.plane_downsample > 1:
        factor = int(fast_cfg.plane_downsample)
        try:
            cfg.plane.du *= factor
            cfg.plane.dv *= factor
            if diag_level >= 1:
                print(
                    f"[run] fast mode downsampling plane: "
                    f"du→{cfg.plane.du}, dv→{cfg.plane.dv} (factor={factor})"
                )
        except Exception as exc:
            if diag_level >= 1:
                print(f"[run] fast mode plane downsample failed: {exc!r}")
                
    # ---- SBP engine override ------------------------------------------
    # If [fast].sbp_engine is explicitly set, use it. Otherwise, when
    # [run].fast = true and the user left run.sbp_engine at its default
    # ("scan"), we implicitly switch to "poly" for speed.
    if getattr(fast_cfg, "sbp_engine", None) is not None:
        cfg.run.sbp_engine = fast_cfg.sbp_engine
    else:
        # Implicit default: fast mode prefers "poly" unless the user chose
        # something else in [run].
        if getattr(cfg.run, "sbp_engine", "scan") == "scan":
            cfg.run.sbp_engine = "poly"



def _iter_source_events(cfg: Config) -> Iterable[Event]:
    """
    Unified entry point for real-data event sources.

    PHITS usrdef  and  ROOT NOVO DDAQ  both use the raw-event → shaping → typed
    pipeline (Stage 1 in run_pipeline).  Therefore `_iter_source_events` should
    *not* call adapter.iter_events() for these formats.

    Instead, run_pipeline() handles:
        adapter.iter_raw_events(...) →
        hit-level filters →
        is_reconstructable →
        shape_events_for_cones →
        shaped_to_typed_events

    This helper is now used ONLY when an adapter truly yields already-typed
    Event objects (future formats).  Today, that's none.
    """

    # Future: if cfg.io.input_format == "hdf5_ngimager", then re-hydrate typed
    # events here.  For now, RAW formats are handled entirely inside Stage 1,
    # and `_iter_source_events` is bypassed.
    adapter_cfg: Dict[str, object] = dict(cfg.io.adapter)

    det_cfg = getattr(cfg, "detectors", None)
    if det_cfg is not None:
        mat_map = getattr(det_cfg, "material_map", None)
        if mat_map and "material_map" not in adapter_cfg:
            adapter_cfg["material_map"] = mat_map

        default_mat = getattr(det_cfg, "default_material", None)
        if default_mat and "default_material" not in adapter_cfg:
            adapter_cfg["default_material"] = default_mat

    adapter = make_adapter(adapter_cfg)

    # IMPORTANT:
    # PHITS usrdef  →  handled by Stage 1 (raw events)
    # ROOT NOVO DDAQ → handled by Stage 1 (raw events)
    if cfg.io.input_format in ("phits_usrdef", "root_novo_ddaq"):
        raise RuntimeError(
            "_iter_source_events() should not be used for PHITS or ROOT. "
            "Stage 1 in run_pipeline() performs raw-event → typed-event shaping."
        )

    # Only for future adapters that yield fully-typed Event objects:
    return adapter.iter_events(str(cfg.io.input_path))



def _build_cones_from_events(
    cfg: Config,
    events: Sequence[Event],
    plane: Plane,
    counters: Dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn events into cone geometry arrays for SBP.

    Returns
    -------
    cone_ids : (N,) uint32
    apex_xyz_cm : (N,3) float32
    axis_xyz : (N,3) float32
    theta_rad : (N,) float32
    species : (N,) uint8
        0 = neutron, 1 = gamma
    recoil_code : (N,) uint8
        0 = unknown / not applicable
        1 = proton recoil (for neutron cones)
        2 = carbon recoil (for neutron cones)
    incident_energy_MeV : (N,) float32
    event_index : (N,) int32
        Row index into the /lm/event_* and /lm/hit_* arrays.
    gamma_hit_order : (N,3) int8
        For gamma cones (species == 1), each row is (i0, i1, i2) giving
        the indices into /lm/hit_*[event_index, :, :] that correspond to
        (first scatter, second scatter, third point). For neutron cones
        (species == 0), entries are (-1, -1, -1).
    """
    lut_registry = build_lut_registry(cfg.energy.lut_paths)
    energy_model = make_energy_strategy(cfg.energy, lut_registry=lut_registry)
    # Prior for cone selection (especially important for gammas and neutron p/C).
    prior = make_prior(cfg.prior.model_dump(), plane)
    # prior = make_prior(cfg.prior.dict(), plane)  # Fallback for older Pydantic

    diag_level = cfg.run.diagnostics_level

    cones: list[Cone] = []
    species_codes: list[int] = []
    recoil_codes: list[int] = []
    incident_energies: list[float] = []
    event_indices: list[int] = []
    gamma_orders: list[tuple[int, int, int]] = []

    for j, ev in enumerate(events):
        # enforce time ordering & sanity without crashing the whole run
        try:
            ev = ev.ordered()  # returns same type
            ev.validate(strict=False)
        except Exception as exc:
            if j < 5 and diag_level >= 2:
                print(f"[stage3] Skipping event {j} during ordered/validate: {exc}")
            continue

        # ---- Species-aware cone building ----
        if isinstance(ev, NeutronEvent):
            if not cfg.run.neutrons:
                continue
            species_char = "n"
            species_code = 0  # 0 = neutron in HDF
        elif isinstance(ev, GammaEvent):
            if not cfg.run.gammas:
                continue
            species_char = "g"
            species_code = 1  # 1 = gamma in HDF
        else:
            # Unknown event type – ignore for now
            continue

        _inc(counters, "events_cone_build_attempted", 1)
        if species_char == "n":
            _inc(counters, "events_cone_build_attempted_n", 1)
        elif species_char == "g":
            _inc(counters, "events_cone_build_attempted_g", 1)

        try:
            if species_char == "n":
                # Neutrons: build proton vs carbon recoil hypotheses and,
                # when a plane/prior are available, select the most plausible one.
                cone, recoil_code, En = build_cone_from_neutron(
                    ev,
                    energy_model,
                    plane=plane,
                    prior=prior,
                    force_proton=cfg.energy.force_proton_recoils,
                    return_meta=True,
                )
                incident_energy = float(En)
                # Neutrons do not have a 3-hit Compton ordering; use sentinel.
                hit_order = (-1, -1, -1)
            else:
                # Gammas: Compton-based cone construction (uses Hit.L
                # as deposited energy; energy_model is passed for future
                # gamma LUT support even if not used now).
                cone, Eg, perm = build_cone_from_gamma(
                    ev,
                    energy_model,
                    plane=plane,
                    prior=prior,
                    return_meta=True,
                    return_perm=True,
                )
                recoil_code = 0  # not applicable for gammas
                incident_energy = float(Eg)
                hit_order = tuple(int(i) for i in perm)
        except Exception as exc:
            _inc(counters, "events_cone_build_failed", 1)
            if species_char == "n":
                _inc(counters, "events_cone_build_failed_n", 1)
            elif species_char == "g":
                _inc(counters, "events_cone_build_failed_g", 1)

            if j < 5 and diag_level >= 2:
                print(f"[stage3] Failed to build cone from event {j}: {exc}")
            continue

        # Δθ cone-level filter (optional; configured via [filters])
        species_label = "neutron" if species_char == "n" else "gamma"
        if not passes_delta_theta_cut(
            cone=cone,
            species=species_label,
            plane=plane,
            prior=prior,
            filters=cfg.filters.cones,
            counters=counters,
            incident_energy_MeV=incident_energy,
        ):
            # This cone is rejected by the Δθ cut.
            continue

        # Success: keep this cone
        _inc(counters, "events_cone_build_success", 1)
        if species_char == "n":
            _inc(counters, "events_cone_build_success_n", 1)
        elif species_char == "g":
            _inc(counters, "events_cone_build_success_g", 1)

        cones.append(cone)
        species_codes.append(species_code)
        recoil_codes.append(int(recoil_code))
        incident_energies.append(incident_energy)
        event_indices.append(j)
        gamma_orders.append(hit_order)

        # Fast-mode: optional conservative cap on number of cones
        if cfg.run.max_cones is not None:
            if len(cones) >= cfg.run.max_cones:
                if diag_level >= 1:
                    print(f"[stage3] Reached max_cones={cfg.run.max_cones}, stopping cone build.")
                break

    if not cones:
        if diag_level >= 1:
            print("[stage3] No cones were successfully built.")
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 3), dtype=np.int8),
        )

    cone_ids = np.arange(len(cones), dtype=np.uint32)
    apex_xyz_cm = np.stack([c.apex for c in cones], axis=0).astype(np.float32)
    axis_xyz = np.stack([c.dir for c in cones], axis=0).astype(np.float32)
    theta_rad = np.array([c.theta for c in cones], dtype=np.float32)
    species_arr = np.array(species_codes, dtype=np.uint8)
    recoil_arr = np.array(recoil_codes, dtype=np.uint8)
    incident_energy_arr = np.array(incident_energies, dtype=np.float32)
    event_index_arr = np.array(event_indices, dtype=np.int32)
    gamma_hit_order_arr = np.array(gamma_orders, dtype=np.int8)

    n_total = int(len(cones))
    n_n = int(np.count_nonzero(species_arr == 0))
    n_g = int(np.count_nonzero(species_arr == 1))
    _inc(counters, "cones_kept", n_total)
    _inc(counters, "cones_kept_n", n_n)
    _inc(counters, "cones_kept_g", n_g)

    if diag_level >= 1:
        print(
            "[stage3] Built {total} cones from {n_ev} events "
            "(neutron={n_n}, gamma={n_g})".format(
                total=n_total,
                n_ev=len(events),
                n_n=n_n,
                n_g=n_g,
            )
        )

    return (
        cone_ids,
        apex_xyz_cm,
        axis_xyz,
        theta_rad,
        species_arr,
        recoil_arr,
        incident_energy_arr,
        event_index_arr,
        gamma_hit_order_arr,
    )


def run_pipeline(
    cfg_path: str,
    *,
    fast: Optional[bool] = None,
    list_mode: Optional[bool] = None,
    neutrons: Optional[bool] = None,
    gammas: Optional[bool] = None,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    plot_label: Optional[str] = None,
) -> Path:
    """
    Orchestrate the full pipeline from a TOML config file.

    CLI flags (--fast/--list/--neutrons/--no-neutrons/--gammas/--no-gammas)
    override the corresponding [run] fields when not None.

    Parameters
    ----------
    cfg_path : str
        Path to TOML configuration file.
    input_path : str, optional
        When provided, overrides [io].input_path from the TOML file.
    output_path : str, optional
        When provided, overrides [io].output_path from the TOML file.
    plot_label : str, optional
        When provided, overrides [run].plot_label (used for HDF5 and
        visualization annotations).

    Returns
    -------
    Path to written HDF5 file.
    """
    #cfg_path = str(cfg_path)
    cfg = load_config(cfg_path)

    # ---- apply CLI overrides on top of TOML ----
    if fast is not None:
        cfg.run.fast = fast
    if list_mode is not None:
        cfg.run.list = list_mode
    if neutrons is not None:
        cfg.run.neutrons = neutrons
    if gammas is not None:
        cfg.run.gammas = gammas

    if input_path is not None:
        cfg.io.input_path = input_path
    if output_path is not None:
        cfg.io.output_path = output_path
    if plot_label is not None:
        cfg.run.plot_label = plot_label
    
    # Conveniences
    diag_level = cfg.run.diagnostics_level
    verbose = diag_level >= 2

    # Basic logging
    if diag_level >= 1:
        print(f"[run] config = {cfg_path}")
        print(f"[run] neutrons={cfg.run.neutrons} gammas={cfg.run.gammas} "
              f"fast={cfg.run.fast} list={cfg.run.list}")
        print(f"[run] input={cfg.io.input_path} -> output={cfg.io.output_path}")
        if getattr(cfg.run, "plot_label", None):
            print(f"[run] plot_label={cfg.run.plot_label!r}")

    # Apply fast-mode overrides (if run.fast is true)
    _apply_fast_overrides(cfg, diag_level=diag_level)
    
    # Imaging plane
    plane = Plane.from_cfg(
        cfg.plane.origin,
        cfg.plane.normal,
        cfg.plane.u_min,
        cfg.plane.u_max,
        cfg.plane.du,
        cfg.plane.v_min,
        cfg.plane.v_max,
        cfg.plane.dv,
        eu=cfg.plane.u_axis,
        ev=cfg.plane.v_axis,
    )

    # HDF5 output
    out_path = Path(cfg.io.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = write_init(str(out_path), cfg_path, cfg, plane)
    
    
    # If this is a NOVO DDAQ ROOT run, try to capture the 'meta' TTree.
    if cfg.io.input_format == "root_novo_ddaq":
        adapter_cfg_meta: Dict[str, object] = dict(cfg.io.adapter)

        adapter_for_meta = make_adapter(adapter_cfg_meta)
        extractor = getattr(adapter_for_meta, "read_meta_tree", None)

        if callable(extractor):
            try:
                meta_dict = extractor(str(cfg.io.input_path))
            except Exception as exc:
                if diag_level >= 1:
                    print(f"[meta] Failed to read ROOT meta tree: {exc}")
            else:
                if meta_dict:
                    write_root_novo_meta(f, meta_dict)
                    if diag_level >= 1:
                        print("[meta] ROOT meta tree written to /meta/root_novo_ddaq")
        else:
            if diag_level >= 1:
                print("[meta] Adapter does not support ROOT meta tree extraction; skipping")
    
    
    # Shared counters for this run
    counters: Dict[str, int] = {}

    # ---- Stage 1: adapter → raw events → hit-level filters → is_reconstructable ----
    if cfg.io.input_format in ("phits_usrdef", "root_novo_ddaq"):
        from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
        from ngimager.filters.to_typed_events import shaped_to_typed_events
        from ngimager.filters.hit_filters import apply_hit_filters, is_reconstructable

        if diag_level >= 1:
            print("\n[stage1] Raw events → hits")
            print("[stage1] Using staged path: raw events → hits → shaped → typed")

        # Build adapter config, injecting detector-level material info.
        adapter_cfg: Dict[str, object] = dict(cfg.io.adapter)

        det_cfg = getattr(cfg, "detectors", None)
        if det_cfg is not None:
            mat_map = getattr(det_cfg, "material_map", None)
            if mat_map and "material_map" not in adapter_cfg:
                adapter_cfg["material_map"] = mat_map

            default_mat = getattr(det_cfg, "default_material", None)
            if default_mat and "default_material" not in adapter_cfg:
                adapter_cfg["default_material"] = default_mat
        
        # --- Geometry: detector-frame → world-frame + per-detector corrections ----
        geom_cfg = getattr(det_cfg, "geometry", None) if det_cfg is not None else None

        # Global frame transform
        frame_cfg = getattr(geom_cfg, "frame", None) if geom_cfg is not None else None
        origin_cm = [0.0, 0.0, 0.0]
        rotation_deg = [0.0, 0.0, 0.0]
        use_global_transform = False
        if frame_cfg is not None:
            origin_cm = getattr(frame_cfg, "origin_cm", origin_cm)
            rotation_deg = getattr(frame_cfg, "rotation_deg", rotation_deg)
            if not is_identity_transform(origin_cm, rotation_deg):
                use_global_transform = True
                if diag_level >= 1:
                    print(
                        "[stage1] Using global detector→world transform: "
                        f"origin_cm={origin_cm}, rotation_deg={rotation_deg}"
                    )

        # Per-detector transforms: id → (origin_cm, rotation_deg)
        per_det_transforms: Dict[int, tuple[list[float], list[float]]] = {}
        if geom_cfg is not None:
            det_list = getattr(geom_cfg, "detectors", []) or []
            for det_entry in det_list:
                try:
                    det_id = int(det_entry.id)
                except Exception:
                    continue
                o = getattr(det_entry, "origin_cm", [0.0, 0.0, 0.0])
                r = getattr(det_entry, "rotation_deg", [0.0, 0.0, 0.0])
                # Skip entries that are effectively identity transforms
                if is_identity_transform(o, r):
                    continue
                per_det_transforms[det_id] = (o, r)

        use_per_det_transforms = bool(per_det_transforms)
        if use_per_det_transforms and diag_level >= 1:
            print(
                "[stage1] Per-detector transforms configured for "
                f"{len(per_det_transforms)} detector IDs"
            )

        
        adapter = make_adapter(adapter_cfg)

        raw_events_after_filters = []

        for ev in adapter.iter_raw_events(str(cfg.io.input_path)):
            hits = list(ev.get("hits", []))

            # Apply per-detector corrections first (local → detector frame)
            if use_per_det_transforms and hits:
                for h in hits:
                    t_cfg = per_det_transforms.get(h.det_id)
                    if t_cfg is None:
                        continue
                    o_det, r_det = t_cfg
                    # apply_rigid_transform accepts (..., 3); we pass a single row
                    h.r = apply_rigid_transform(h.r[None, :], o_det, r_det)[0]

            # Apply global detector-frame → world-frame transform
            if use_global_transform and hits:
                pts = np.stack([h.r for h in hits], axis=0)
                pts_world = apply_rigid_transform(pts, origin_cm, rotation_deg)
                for h, r_new in zip(hits, pts_world):
                    h.r = r_new

            # Normalize event_type to 'n' / 'g' where possible
            et_raw = str(ev.get("event_type", "")).lower()
            if et_raw.startswith("n"):
                et = "n"
            elif et_raw.startswith("g"):
                et = "g"
            else:
                et = None

            counters["raw_events_total"] = counters.get("raw_events_total", 0) + 1

            # ---- Stage 1.5: Apply Hit-level filters
            filtered_hits = apply_hit_filters(
                hits,
                cfg.filters.hits,
                counters,
                particle_type=et,
            )

            # Early reconstructability decision (also updates *_unreconstructable counters)
            if not is_reconstructable(filtered_hits, cfg.filters.events, counters, event_type=et):
                continue

            if not filtered_hits:
                # Should be caught by is_reconstructable, but guard anyway
                continue

            ev2 = dict(ev)
            ev2["hits"] = filtered_hits
            if et is not None:
                ev2["event_type"] = et
            raw_events_after_filters.append(ev2)

        if diag_level >= 1:
            print(
                "[stage1] raw_events_total={total} "
                "raw_events_after_filters={surv} "
                "raw_events_rejected_unreconstructable={rej}".format(
                    total=counters.get("raw_events_total", 0),
                    surv=len(raw_events_after_filters),
                    rej=counters.get("raw_events_rejected_unreconstructable", 0),
                )
            )


        # ---- Stage 2: Hits → shaped events → typed events ----
        if diag_level >= 1:
            print("\n[stage2] Hits → shaped events → typed events")

        # Shaper configuration:
        #   - PHITS: use default policies (time-ascending for both species).
        #   - ROOT NOVO DDAQ: keep neutrons time-ordered, but pick the
        #     brightest 3 gamma hits when multiplicity > 3.
        shape_cfg = ShapeConfig()
        if cfg.io.input_format == "root_novo_ddaq":
            # ROOT neutrons: enforce time-order like legacy
            shape_cfg.neutron_policy = "time_asc"
            # ROOT gammas: keep brightest-gamma rule
            shape_cfg.gamma_policy = "energy_desc"

        shaped_events, shape_diag = shape_events_for_cones(
            raw_events_after_filters,
            shape_cfg,
            counters=counters,
        )

        if diag_level >= 1:
            print(
                "[stage2] shaper: total_events_in={total} "
                "shaped_n={sn} shaped_g={sg}".format(
                    total=shape_diag.total_events,
                    sn=shape_diag.shaped_neutron,
                    sg=shape_diag.shaped_gamma,
                )
            )


        if diag_level >= 2 and shaped_events:
            print(f"[stage2] Example shaped events (up to first 3):")
            for se in shaped_events[:3]:
                ts = [h.t_ns for h in se.hits]
                Ls = [h.L for h in se.hits]
                types = [h.type for h in se.hits]
                print(
                    f"    species={se.species} "
                    f"n_hits={len(se.hits)} "
                    f"types={types} "
                    f"t_ns={ts} "
                    f"L={Ls}"
                )

        events = shaped_to_typed_events(
            shaped_events,
            order_time=True,
        )

        # ---- Typed events diagnostics (Stage: hits → shaped → typed) ----
        # Species breakdown based on the typed-event objects themselves.
        # This does not assume any particular event-level filters yet.
        from ngimager.physics.events import NeutronEvent, GammaEvent
        n_n = sum(isinstance(ev, NeutronEvent) for ev in events)
        n_g = sum(isinstance(ev, GammaEvent) for ev in events)
        counters["events_typed_total"] = n_n + n_g
        counters["events_typed_n"] = n_n
        counters["events_typed_g"] = n_g
        # Placeholder for future event-level rejections (event filters)
        # so that a later filter stage can do:
        #   counters["events_rejected_filters"] = ...
        events_rejected = counters.get("events_rejected_filters", 0)

        if diag_level >= 1:
            print(
                "[stage2] typed_total={total} "
                "typed_n={n} typed_g={g} "
                "events_rejected_filters={rej}".format(
                    total=len(events),
                    n=n_n,
                    g=n_g,
                    rej=events_rejected,
                )
            )
            
    else:
        # For non-PHITS sources, keep the existing direct typed-event path.
        events = list(_iter_source_events(cfg))


    # ---- Stage 2.5: Apply Event-level filters (e.g. ToF windows) ----
    events = apply_event_filters(
        events,
        cfg.filters.events,
        counters,
    )

    if diag_level >= 1:
        print(
            "[stage2] events_total_for_filters={etf} "
            "events_after_filters={eaf} "
            "events_rejected_filters={er}".format(
                etf=counters.get("events_total_for_filters", 0),
                eaf=counters.get("events_after_filters", 0),
                er=counters.get("events_rejected_filters", 0),
            )
        )

    # Existing diagnostics on typed events
    if diag_level >= 1:
        print(f"[stage2] Got {len(events)} events")
    if events:
        first = events[0]
        h1 = getattr(first, "h1", None)
        h2 = getattr(first, "h2", None)
        if diag_level >= 2:
            print(f"[stage2] First event type: {type(first).__name__}")
            print(f"[stage2] First event h1: {h1!r}")
            print(f"[stage2] First event h2: {h2!r}")
            if h1 is not None:
                print(f"[stage2] h1.r = {getattr(h1, 'r', None)}, t_ns={h1.t_ns}, L={h1.L}")
            if h2 is not None:
                print(f"[stage2] h2.r = {getattr(h2, 'r', None)}, t_ns={h2.t_ns}, L={h2.L}")
            for ev in events[:3]:
                species = "n" if isinstance(ev, NeutronEvent) else "g" if isinstance(ev, GammaEvent) else "?"
                hlist = [getattr(ev, name) for name in ("h1", "h2", "h3") if hasattr(ev, name)]
                ts = [h.t_ns for h in hlist]
                Ls = [h.L for h in hlist]
                types = [h.type for h in hlist]
                print(
                    f"    {species}-event "
                    f"n_hits={len(hlist)} "
                    f"types={types} "
                    f"t_ns={ts} "
                    f"L={Ls}"
                )

    # Write to output Per-event / per-hit physics (this links back via /lm/events dataset)
    write_events_hits(f, events)
    
    

    # ---- Stage 3: Typed events → candidate cones → selected cones ----
    if diag_level >= 1:
        print("\n[stage3] Events → candidate cones → selected cones")
    # Cones from events
    (
        cone_ids,
        apex_xyz_cm,
        axis_xyz,
        theta_rad,
        cone_species,
        recoil_code,
        incident_energy_MeV,
        cone_event_index,
        gamma_hit_order,
    ) = _build_cones_from_events(cfg, events, plane, counters)
    
    # --- Build per-event cone survival arrays (event → cone_id) ---
    n_events = len(events)
    event_cone_id = np.full(n_events, -1, dtype=np.int32)
    event_imaged_cone_id = np.full(n_events, -1, dtype=np.int32)  # will be filled in LM block if list-mode

    for cid, ev_idx in zip(cone_ids, cone_event_index):
        # Defensive check; ev_idx should be in [0, n_events)
        if 0 <= int(ev_idx) < n_events:
            # For now, each event yields at most one cone; keep the first if ever that changes.
            if event_cone_id[ev_idx] == -1:
                event_cone_id[ev_idx] = int(cid)

    # Counters and diagnostics for cones
    n_cones = int(len(cone_ids))
    n_n = int(np.count_nonzero(cone_species == 0))
    n_g = int(np.count_nonzero(cone_species == 1))
    # Neutron recoil breakdown
    n_p = int(np.count_nonzero((cone_species == 0) & (recoil_code == 1)))
    n_C = int(np.count_nonzero((cone_species == 0) & (recoil_code == 2)))
    n_unknown = max(0, n_n - n_p - n_C)
    _inc(counters, "cones_n_recoil_proton", n_p)
    _inc(counters, "cones_n_recoil_carbon", n_C)
    _inc(counters, "cones_n_recoil_unknown", n_unknown)
    
    if diag_level >= 1:
        print(
            "[stage3] Built {total} cones "
            "(neutron={n_n}, gamma={n_g})".format(
                total=n_cones,
                n_n=n_n,
                n_g=n_g,
            )
        )
        if n_cones and diag_level >= 2:
            print("[stage3] Example cone apex:", apex_xyz_cm[0])
            print("[stage3] Example cone dir:", axis_xyz[0])
            print("[stage3] Example cone theta[deg]:", np.degrees(theta_rad[0]))
            print(
                "[stage3] Neutron recoil breakdown: "
                f"proton={n_p}, carbon={n_C}, unknown={n_unknown}"
            )

    # Write to output Per-cone geometry + classification
    write_cones(
        f,
        cone_ids,
        apex_xyz_cm,
        axis_xyz,
        theta_rad,
        species=cone_species,
        recoil_code=recoil_code,
        incident_energy_MeV=incident_energy_MeV,
        event_index=cone_event_index,
        gamma_hit_order=gamma_hit_order,
    )

    # ---- Stage 4: Cones → imaging/reconstruction (SPB) ----
    if diag_level >= 1:
        print("\n[stage4] Cones → imaging / reconstruction (SBP)")
    # Choose SBP engine from config
    sbp_engine = getattr(cfg.run, "sbp_engine", "scan")

    # Build Cone objects from the geometry arrays
    cones_for_sbp: list[Cone] = [
        Cone(apex=apex_xyz_cm[i], direction=axis_xyz[i], theta=float(theta_rad[i]))
        for i in range(len(cone_ids))
    ]

    # Partition cones by species for separate images
    cones_n: list[Cone] = []
    cones_g: list[Cone] = []
    for c, s in zip(cones_for_sbp, cone_species):
        if s == 0:
            cones_n.append(c)
        elif s == 1:
            cones_g.append(c)

    img_n = None
    img_g = None
    
    # Projection / ROI configuration (for HDF5 + visualization)
    vis_cfg = getattr(cfg, "vis", None)
    proj_cfg = getattr(vis_cfg, "projections", None) if vis_cfg is not None else None

    projections_enabled = bool(
        getattr(proj_cfg, "enabled", False)
    ) if proj_cfg is not None else False

    roi: tuple[float, float, float, float] | None = None
    if projections_enabled and proj_cfg is not None:
        u0 = getattr(proj_cfg, "roi_u_min_cm", None)
        u1 = getattr(proj_cfg, "roi_u_max_cm", None)
        v0 = getattr(proj_cfg, "roi_v_min_cm", None)
        v1 = getattr(proj_cfg, "roi_v_max_cm", None)
        # Only use ROI if all four bounds are provided
        if None not in (u0, u1, v0, v1):
            roi = (float(u0), float(u1), float(v0), float(v1))


    # --- 4a. Neutron-only image ---
    if cones_n:
        if diag_level >= 1:
            print( "[stage4] Imaging neutrons...")
        
        recon_n = reconstruct_sbp(
            cones=cones_n,
            plane=plane,
            workers=cfg.run.workers,
            chunk_cones=cfg.run.chunk_cones,
            list_mode=False,  # LM handled separately below
            # uncertainty_mode stays at default "off" for now
            progress=cfg.run.progress,
            sbp_engine=sbp_engine,
            use_jit=cfg.run.jit,
        )

        img_n = recon_n.summed.astype(np.float32)
        write_summed(f, "n", img_n)

        if diag_level >= 1:
            print(
                "[stage4] Recon summed image stats (n):",
                "min=", float(img_n.min()),
                "max=", float(img_n.max()),
                "sum=", float(img_n.sum()),
                "shape=", img_n.shape,
            )

    # --- 4b. Gamma-only image ---
    if cones_g:
        if diag_level >= 1:
            print( "[stage4] Imaging gammas...")
            
        recon_g = reconstruct_sbp(
            cones=cones_g,
            plane=plane,
            workers=cfg.run.workers,
            chunk_cones=cfg.run.chunk_cones,
            list_mode=False,  # LM handled separately below
            progress=cfg.run.progress,
            sbp_engine=sbp_engine,
            use_jit=cfg.run.jit,
        )

        img_g = recon_g.summed.astype(np.float32)
        write_summed(f, "g", img_g)

        if diag_level >= 1:
            print(
                "[stage4] Recon summed image stats (g):",
                "min=", float(img_g.min()),
                "max=", float(img_g.max()),
                "sum=", float(img_g.sum()),
                "shape=", img_g.shape,
            )

    # --- 4c. "all" image = n + g (when both exist) ---
    img_all = None
    if img_n is not None and img_g is not None:
        # Both species present: sum them
        img_all = (img_n + img_g).astype(np.float32)
    elif img_n is not None:
        # Only neutrons present
        img_all = img_n
    elif img_g is not None:
        # Only gammas present
        img_all = img_g

    if img_all is not None:
        write_summed(f, "all", img_all)
        if diag_level >= 1:
            print(
                "[stage4] SUMMED Recon summed image stats (all):",
                "min=", float(img_all.min()),
                "max=", float(img_all.max()),
                "sum=", float(img_all.sum()),
                "shape=", img_all.shape,
            )
            
    # --- 4c.1. 1D projections (optional) ------------------------------------
    if projections_enabled:
        if img_n is not None:
            write_projections(f, "n", img_n, roi=roi)
        if img_g is not None:
            write_projections(f, "g", img_g, roi=roi)
        if img_all is not None:
            write_projections(f, "all", img_all, roi=roi)


    # --- 4d. List-mode extras: explicit cone → pixel mapping + event survival ---
    if cfg.run.list and len(cones_for_sbp) > 0:
        lm_cone_pixel_lists: list[tuple[int, np.ndarray]] = []
        # For list-mode runs, we can now also record which cones actually
        # intersected the plane (non-empty pixel sets) on a per-event basis.
        for cid, c, ev_idx in zip(cone_ids, cones_for_sbp, cone_event_index):
            idx = cone_to_indices(c, plane, engine=sbp_engine, n_poly=360, use_jit=cfg.run.jit)  # match SBP default
            if idx is None or idx.size == 0:
                continue
            lm_cone_pixel_lists.append((int(cid), idx))
            # Mark this event as having an imaged cone, if not already set.
            if 0 <= int(ev_idx) < len(event_imaged_cone_id):
                if event_imaged_cone_id[ev_idx] == -1:
                    event_imaged_cone_id[ev_idx] = int(cid)
        write_lm_indices(f, lm_cone_pixel_lists)

    # Store per-event survival table (event → cone, event → imaged cone)
    write_event_cone_survival(f, event_cone_id, event_imaged_cone_id)
    
    # Store counters into the HDF5 file for later inspection
    write_counters(f, counters)

    # Optional diagnostics printout of counters (console only).
    if diag_level >= 1:
        print("\n[diagnostics] Counter summary:")
        for key in sorted(counters.keys()):
            print(f"  {key} = {counters[key]}")
    
    f.close()

    # Optional image export
    if getattr(cfg, "vis", None) and getattr(cfg.vis, "export_png_on_write", False):
        try:
            # Species to render (n/g/all)
            species = getattr(cfg.vis, "species", ["n", "g", "all"])

            # Always include PNG; add extra formats from config (e.g. ["pdf"])
            formats = ["png"]
            extra = getattr(cfg.vis, "extra_formats", [])
            for fmt in extra:
                fmt = str(fmt).lower()
                if fmt and fmt not in formats:
                    formats.append(fmt)

            image_paths = render_summed_images(
                str(out_path),
                species=species,
                filename_pattern=getattr(
                    cfg.vis,
                    "filename_pattern",
                    "{species}_{stem}.{ext}",
                ),
                center_on_plane_center=getattr(cfg.vis, "center_on_plane_center", True),
                flip_vertical=getattr(cfg.vis, "flip_vertical", True),
                axis_units=getattr(cfg.vis, "axis_units", "cm"),
                cmap=getattr(cfg.vis, "cmap", "cividis"),
                formats=formats,
                projections=projections_enabled,
                roi_u_min_cm=roi[0] if roi is not None else None,
                roi_u_max_cm=roi[1] if roi is not None else None,
                roi_v_min_cm=roi[2] if roi is not None else None,
                roi_v_max_cm=roi[3] if roi is not None else None,
                plot_label=getattr(cfg.run, "plot_label", None),
            )

            if cfg.run.diagnostics_level >= 1:
                if image_paths:
                    print("[stage4] Wrote images: " + ", ".join(str(p) for p in image_paths))
                else:
                    print("[stage4] No /images/summed/* datasets found to visualize")
        except Exception as e:
            if cfg.run.diagnostics_level >= 1:
                print(f"[stage4] Image export failed: {e!r}")



    return out_path


# ---------------------------------------------------------------------------
# Unified CLI entry point
# ---------------------------------------------------------------------------

app = typer.Typer(help="Unified ng-imager imaging pipeline (ngimager.pipelines.core)")


@app.command()
def main(
    cfg_path: str = typer.Argument(
        ...,
        help="Path to TOML config file",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Override [run].fast = true (use aggressive fast settings)",
    ),
    list_mode: bool = typer.Option(
        False,
        "--list",
        help="Override [run].list = true (enable list-mode image output)",
    ),
    neutrons: Optional[bool] = typer.Option(
        None,
        "--neutrons / --no-neutrons",
        help="Enable or disable neutron processing; overrides [run].neutrons when set",
    ),
    gammas: Optional[bool] = typer.Option(
        None,
        "--gammas / --no-gammas",
        help="Enable or disable gamma processing; overrides [run].gammas when set",
    ),
    input_path: Optional[str] = typer.Option(
        None,
        "--input-path",
        "-i",
        help="Override [io].input_path from the TOML config.",
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output-path",
        "-o",
        help="Override [io].output_path from the TOML config.",
    ),
    plot_label: Optional[str] = typer.Option(
        None,
        "--plot-label",
        help="Override [run].plot_label (annotation text used in visualization).",
    ),
):
    """
    Run the unified ng-imager pipeline for a single config.
    """
    out_path = run_pipeline(
        cfg_path,
        fast=fast if fast else None,
        list_mode=list_mode if list_mode else None,
        neutrons=neutrons,
        gammas=gammas,
        input_path=input_path,
        output_path=output_path,
        plot_label=plot_label,
    )
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
