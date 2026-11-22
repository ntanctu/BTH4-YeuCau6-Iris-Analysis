"""Small regression tests; use: python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

class TestProject(unittest.TestCase):
    def setUp(self):
        self.raw=pd.read_csv(ROOT/'data'/'iris_uci_original.csv')
        self.clean=pd.read_csv(ROOT/'data'/'iris_cleaned.csv')
        self.stats=json.loads((ROOT/'reports'/'ket_qua_phan_tich.json').read_text(encoding='utf-8'))

    def test_original_rows_and_columns(self):
        self.assertEqual(self.raw.shape,(150,5))
        self.assertEqual(self.raw.loc[34,'petal_width_cm'],0.1)
        self.assertEqual(self.raw.loc[37,'sepal_width_cm'],3.1)

    def test_cleaned_data(self):
        self.assertEqual(self.clean.shape,(150,5))
        self.assertFalse(self.clean.isnull().any().any())
        self.assertEqual(self.clean['species'].value_counts().to_dict(),{'Setosa':50,'Versicolor':50,'Virginica':50})

    def test_model_consistency(self):
        m=self.stats['model']
        self.assertEqual(m['train_rows']+m['test_rows'],150)
        self.assertEqual(m['test_rows'],45)
        self.assertGreaterEqual(m['accuracy_test'],.6)
        self.assertEqual(sum(map(sum,m['cm'])),45)

    def test_deliverables(self):
        self.assertGreater((ROOT/'reports'/'bao_cao_khoa_hoc_iris.pdf').stat().st_size,10000)
        for f in ['01_boxplot_petal.png','02_scatter_petal.png','03_correlation.png','04_confusion.png']:
            self.assertGreater((ROOT/'assets'/f).stat().st_size,1000)

if __name__=='__main__':unittest.main()
