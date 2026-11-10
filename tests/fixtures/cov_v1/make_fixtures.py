"""Regenerate the v1 covariance fixtures: good_ref.txt (3 columns, %.4e, like LSS_Reader
output) and good.cov.json (full precision, spec 7.1 schema, one 'stitch' multiplicative
term).  N = 12, K = 3, one 0.5/0.5 overlap row.   python make_fixtures.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 12


def build():
    x = 0.02 * (1.0 + 0.1 * np.arange(N)) + 1e-5 * np.sqrt(np.arange(N) + 2.0)
    y = np.exp(-x * 7.3) * (1.0 + 0.0123456789 * np.arange(N))
    sigma = 0.03 * y * (1.0 + 0.0137 * np.arange(N))
    w = np.zeros((N, 3))
    w[0:4, 0] = 1.0
    w[4] = [0.5, 0.5, 0.0]          # the overlap row
    w[5:8, 1] = 1.0
    w[8, 1] = 0.5
    w[8, 2] = 0.5
    w[9:, 2] = 1.0
    cov = np.array([[0, 0, 0], [0, 4e-4, 4e-4], [0, 4e-4, 9e-4]], float)
    return x, y, sigma, w, cov


def main():
    x, y, sigma, w, cov = build()
    np.savetxt(os.path.join(HERE, 'good_ref.txt'), np.c_[x, y, sigma], fmt='%.4e')
    doc = {
        'format': 'xmodfit-covariance', 'version': 1,
        'producer': {'name': 'LSS_Reader', 'version': 'fixture'},
        'data_file': 'good_ref.txt',
        'x': x.tolist(), 'y': y.tolist(), 'sigma_meas': sigma.tolist(),
        'terms': [{
            'name': 'stitch', 'scope': 'dataset', 'kind': 'multiplicative',
            'labels': ['S12', 'S13', 'S14'], 'weights': w.tolist(), 'cov': cov.tolist(),
            'provenance': {'fac': [1.0, 0.0213, 0.0118], 'facerr': [0.0, 4.3e-4, 2.6e-4]},
        }],
    }
    with open(os.path.join(HERE, 'good.cov.json'), 'w') as f:
        json.dump(doc, f, indent=1)


if __name__ == '__main__':
    main()
