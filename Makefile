.PHONY: test eval eval-stress data
test:        ; PYTHONPATH=src python -m unittest discover -s tests -t .
eval:        ; PYTHONPATH=src python -m scilit.eval.run_eval --repeats 3
eval-stress: ; PYTHONPATH=src python -m scilit.eval.run_eval --repeats 3 --k 1
data:        ; python scripts/make_sample_data.py
