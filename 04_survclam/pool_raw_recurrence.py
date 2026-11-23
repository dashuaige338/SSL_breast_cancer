"""Calculate pooled recurrence concordance from untransformed held-out risks."""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

ROOT = Path(__file__).resolve().parent / 'eval_results'
OUT = ROOT / 'raw_pooled_recurrence'


def main():
    results, predictions = [], {}
    for model in ['hoptimus', 'virchow']:
        source = ROOT / f'EVAL_poet_breast_recurrence_{model}_hospitals_20x_s1_cv_test'
        frames = []
        for fold in range(28):
            frame = pd.read_csv(source / f'fold_{fold}.csv')
            frame['fold'] = fold
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        assert data.slide_id.notna().all() and not data.slide_id.duplicated().any()
        assert np.isfinite(data[['time', 'event', 'risk_score']].to_numpy()).all()
        assert data.event.isin([0, 1]).all()
        percentile = data.groupby('fold').risk_score.rank(method='average', pct=True)
        ranked_score = concordance_index(data.time, -percentile, data.event)
        previous = pd.read_csv(ROOT / 'pooled_oof_km_with_cindex_summary.csv')
        label = 'H-optimus-0' if model == 'hoptimus' else 'Virchow'
        expected = previous.loc[previous.model.eq(label), 'pooled_oof_cindex'].iloc[0]
        assert np.isclose(ranked_score, expected, atol=1e-12)
        predictions[model] = data
        results.append(dict(model=model, folds=28, rows=len(data), events=int(data.event.sum()),
                            raw_pooled_cindex=concordance_index(data.time, -data.risk_score, data.event),
                            percentile_pooled_cindex=ranked_score))
    OUT.mkdir(exist_ok=True)
    for model, data in predictions.items():
        data.to_csv(OUT / f'{model}_pooled_raw_predictions.csv', index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / 'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
