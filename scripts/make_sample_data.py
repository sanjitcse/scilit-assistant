"""Generate the SYNTHETIC sample corpus + gold labels. Every paper, system and number is fictional.

Usage: python scripts/make_sample_data.py
Writes: data/sample_corpus.jsonl, eval/datasets/qa.jsonl, eval/datasets/extraction.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

P = lambda **k: k  # noqa: E731
PAPERS = [
    P(id="P001", name="GlacierNet", task="glacier outline segmentation from multispectral satellite imagery",
      blurb="a convolutional encoder-decoder with a boundary-aware loss", dataset="AlpineSat-12k",
      training="a boundary-aware loss and random tile cropping", metric="IoU", value="0.81",
      baseline="U-Net", bvalue="0.74", component="the boundary-aware loss",
      limitation="cloud cover above 40% in a tile degrades outlines", kw="cloud cover"),
    P(id="P002", name="PolyFold-Lite", task="fast protein loop modeling",
      blurb="a lightweight diffusion model over backbone torsion angles", dataset="LoopBench",
      training="torsion-angle noise scheduling", metric="RMSD", value="1.9 Å",
      baseline="Rosetta KIC", bvalue="2.6 Å", component="the torsion noise schedule",
      limitation="loops longer than 20 residues are modeled poorly", kw="20 residues"),
    P(id="P003", name="CardioTrace", task="arrhythmia detection from single-lead ECG",
      blurb="a one-dimensional residual network with attention pooling", dataset="HeartWave-5",
      training="attention pooling over beat windows", metric="macro F1", value="0.93",
      baseline="ResNet-18", bvalue="0.88", component="attention pooling",
      limitation="all recordings come from a single hospital", kw="single hospital"),
    P(id="P004", name="SoilSense", task="soil organic carbon prediction from spectra",
      blurb="a partial-least-squares model with learned spectral preprocessing", dataset="SoilSpec-Europe",
      training="learned Savitzky-Golay preprocessing", metric="R2", value="0.87",
      baseline="PLSR", bvalue="0.79", component="learned preprocessing",
      limitation="only lab-dried samples were measured", kw="lab-dried"),
    P(id="P005", name="KernelTab", task="small-sample tabular classification",
      blurb="a learned-kernel support vector classifier", dataset="TabSmall-40",
      training="marginal-likelihood kernel tuning", metric="accuracy", value="0.76",
      baseline="gradient boosting", bvalue="0.71", component="kernel tuning",
      limitation="training cost grows cubically with sample count", kw="cubically"),
    P(id="P006", name="ReefWatch", task="coral bleaching detection in drone video",
      blurb="a temporal detector with optical-flow stabilization", dataset="ReefDrone-3k",
      training="optical-flow stabilization of frames", metric="mAP", value="0.69",
      baseline="YOLOv8", bvalue="0.61", component="optical-flow stabilization",
      limitation="turbid water reduces detection quality", kw="turbid"),
    P(id="P007", name="SynthPlan", task="single-step retrosynthesis prediction",
      blurb="a template-free transformer with reaction-class conditioning", dataset="USPTO-Mini",
      training="reaction-class conditioning", metric="top-1 accuracy", value="52.4%",
      baseline="RetroTemplate", bvalue="47.1%", component="reaction-class conditioning",
      limitation="stereochemistry is often predicted incorrectly", kw="stereochemistry"),
    P(id="P008", name="GustCast", task="wind gust nowcasting",
      blurb="a convolutional LSTM over radar and station data", dataset="GustArchive-DE",
      training="radar-station fusion", metric="MAE", value="1.4 m/s",
      baseline="persistence forecasting", bvalue="2.0 m/s", component="radar-station fusion",
      limitation="forecasts degrade in complex terrain", kw="complex terrain"),
]

UNANSWERABLE = [
    "What F1 score does OrbitalNet reach on the SkyMap dataset?",
    "Which dataset was NeuroLens trained on?",
    "What limitation of DeepGlacier do the authors report?",
]


def paper_record(p):
    n = p["name"]
    return {
        "paper_id": p["id"],
        "title": f"{n}: {p['task'].capitalize()} (synthetic)",
        "doi": f"10.0000/synthetic.{p['id'].lower()}",
        "sections": {
            "abstract": f"We study {p['task']}. We introduce {n}, {p['blurb']}.",
            "methods": (f"We train {n} on the {p['dataset']} dataset using {p['training']}. "
                        "All models are trained with a fixed random seed and evaluated on a held-out test split."),
            "results": (f"On {p['dataset']}, {n} achieves {p['metric']} of {p['value']}, compared with "
                        f"{p['bvalue']} for {p['baseline']}. Removing {p['component']} lowers {p['metric']} "
                        "in ablation studies. Results are averaged over five runs."),
            "limitations": (f"A key limitation of {n} is that {p['limitation']}. "
                            "Addressing this is left to future work."),
        },
    }


def main():
    papers = [paper_record(p) for p in PAPERS]
    qa = []
    for p in PAPERS:
        n, i = p["name"], p["id"]
        qa += [
            dict(question=f"What {p['metric']} does {n} achieve on {p['dataset']}?", answerable=True,
                 gold_keywords=[p["value"]], evidence=[f"{i}:results"], type="result"),
            dict(question=f"Which dataset was {n} trained on?", answerable=True,
                 gold_keywords=[p["dataset"]], evidence=[f"{i}:methods"], type="method"),
            dict(question=f"What limitation of {n} do the authors report?", answerable=True,
                 gold_keywords=[p["kw"]], evidence=[f"{i}:limitations"], type="limitation"),
        ]
    for p in PAPERS:  # harder slice: the paper is described by its task, not named
        i = p["id"]
        qa += [
            dict(question=f"Which dataset was used for {p['task']}?", answerable=True,
                 gold_keywords=[p["dataset"]], evidence=[f"{i}:methods"], type="method_by_task"),
            dict(question=f"What limitation is reported for {p['task']}?", answerable=True,
                 gold_keywords=[p["kw"]], evidence=[f"{i}:limitations"], type="limitation_by_task"),
        ]
    qa += [dict(question=q, answerable=False, gold_keywords=[], evidence=[], type="unanswerable") for q in UNANSWERABLE]
    for j, row in enumerate(qa, 1):
        row["id"] = f"Q{j:03d}"
    extraction = [
        {"paper_id": p["id"], "fields": {
            "task": p["task"], "dataset": p["dataset"], "metric": p["metric"], "value": p["value"],
            "baseline": p["baseline"], "baseline_value": p["bvalue"], "limitation": p["limitation"]}}
        for p in PAPERS
    ]
    (ROOT / "eval/datasets").mkdir(parents=True, exist_ok=True)
    for path, rows in [("data/sample_corpus.jsonl", papers), ("eval/datasets/qa.jsonl", qa),
                       ("eval/datasets/extraction.jsonl", extraction)]:
        with open(ROOT / path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(papers)} papers, {len(qa)} questions, {len(extraction)} extraction records")


if __name__ == "__main__":
    main()
