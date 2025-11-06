This directory preserves the legacy monolithic imaging script and its settings for reference and comparison.  It was developed for imaging analysis, both in simulation and experimental, for the NOVO project. As needs evolved, experimental contexts changed, core static assumptions made variable, and various methodoligies tested, this code grew quite fragile and difficult to maintain.

The general idea was that this code, `expNOVO_imager_legacy.py`, could be used as an "all-in-one" processing tool to convert experimental/simulated coincident event data into cones and then into images using simple back projection.  When executing it on the command line, it would be provided with an input file (pickle file of simulated data or a experimental ROOT file) and optionally a settings configuration file (defaulting to `image_settings.txt`, two examples provided here, which ultimately consists of raw python code to be run via `exec()` calls to overwrite the hard-coded defaults) and a number of processors to use in parallelized imaging of projected cones.

```commandline
python expNOVO_imager_legacy.py "example_phits_simulation_dir\imaging_data_records.pickle" -p 2

python "expNOVO_imager_legacy.py" autoSorted_coinc_detector_DT-14p8MeV_000006_fastmode.pickle -s "C:\path\to\image_settings.txt" -o "C:\path\to\output\dir" -p 3 -f
```

The modern `ng-imager` library here reimplements its physics and imaging pipeline in a modular, testable way.