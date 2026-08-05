# Ghost Scale Simulation — convenience targets.
# Version runners live in runners/ ; run_all.py is the original E1-E34 programme.
# On Windows without `make`, use the equivalent: python run_all.py

PYTHON ?= python
WORKERS ?=

.PHONY: all quick test invariants nulls gates soundingline validate figures clean e1 e2 e3 e4 e5 e6

all:            ## run all six experiments + tests at full spec scale
	$(PYTHON) run_all.py $(if $(WORKERS),--workers $(WORKERS),)

quick:          ## fast smoke-scale run of everything
	$(PYTHON) run_all.py --quick $(if $(WORKERS),--workers $(WORKERS),)

validate:       ## run the validation pass (V-1 .. V-9), writes results/validation/
	$(PYTHON) runners/run_validation.py $(if $(WORKERS),--workers $(WORKERS),)

test:           ## run the full test suite (invariants + nulls)
	$(PYTHON) -m pytest -q

gates:          ## standing controls + metamorphic relations only (fast; see docs/METHODS.md)
	$(PYTHON) -m pytest -q tests/test_gates.py tests/test_metamorphic.py

soundingline:   ## run the batch of tests another project asked for, at full scale
	$(PYTHON) runners/run_soundingline.py

invariants:     ## model-invariant tests only (Spec §10)
	$(PYTHON) -m pytest tests/test_model_invariants.py -q

nulls:          ## null-condition tests only (Spec §9)
	$(PYTHON) -m pytest tests/test_nulls.py -q

figures:        ## redraw every chart from committed verdict files
	$(PYTHON) scripts/rebuild_figures.py
	$(PYTHON) scripts/make_walkthrough_plates.py
	$(PYTHON) scripts/make_ghost_scale_pair.py

e1:; $(PYTHON) -m ghostscale.experiments.e1_crash $(if $(WORKERS),--workers $(WORKERS),)
e2:; $(PYTHON) -m ghostscale.experiments.e2_variance $(if $(WORKERS),--workers $(WORKERS),)
e3:; $(PYTHON) -m ghostscale.experiments.e3_titration $(if $(WORKERS),--workers $(WORKERS),)
e4:; $(PYTHON) -m ghostscale.experiments.e4_trust_exploit $(if $(WORKERS),--workers $(WORKERS),)
e5:; $(PYTHON) -m ghostscale.experiments.e5_precision_baseline $(if $(WORKERS),--workers $(WORKERS),)
e6:; $(PYTHON) -m ghostscale.experiments.e6_corpus_corruption $(if $(WORKERS),--workers $(WORKERS),)

clean:          ## remove generated CSVs and figures
	$(PYTHON) -c "import pathlib,glob; [pathlib.Path(p).unlink() for p in glob.glob('results/*.csv')+glob.glob('figures/*.png')]"
