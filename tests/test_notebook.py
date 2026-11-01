import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.notebook = json.loads((self.root / 'grounding_dino_auto_label.ipynb').read_text())

    def test_valid_clean_notebook_structure(self):
        self.assertEqual(self.notebook['nbformat'], 4)
        self.assertEqual(self.notebook['metadata']['kernelspec']['language'], 'python')
        cells = self.notebook['cells']
        self.assertTrue(any(cell['cell_type'] == 'markdown' for cell in cells))
        code = [cell for cell in cells if cell['cell_type'] == 'code']
        self.assertTrue(code)
        for cell in code:
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'], [])
            self.assertIsInstance(cell['source'], list)

    def test_colab_directory_magics_are_restored_in_order(self):
        source = ''.join(''.join(cell['source']) for cell in self.notebook['cells']
                         if cell['cell_type'] == 'code')
        self.assertTrue(source.startswith('%cd /content\n'))
        self.assertLess(source.index('%cd GroundingDINO'), source.index('!git checkout'))
        self.assertLess(source.index('%cd weights'), source.index('!wget'))
        self.assertLess(source.index('%cd /content/GroundingDINO'), source.index('from groundingdino.util.inference'))
        self.assertNotIn('# %cd ', source)
        self.assertIn('57535c5a79791cb76e36fdb64975271354f10251', source)

    def test_readme_links_real_notebook_not_renaming_instructions(self):
        readme = (self.root / 'README.md').read_text()
        self.assertIn('blob/main/grounding_dino_auto_label.ipynb', readme)
        self.assertNotIn('after renaming to `.ipynb`', readme)


if __name__ == '__main__':
    unittest.main()
