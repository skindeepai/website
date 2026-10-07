"""Regression checks against the portable, real-trial source snapshot."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from context_trial_import import CONFIG, RESULT, ROOT, preserve_history, project_result, read_json, validate_bundle
from refresh_long_context import score


class ContextTrialImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(CONFIG)
        cls.data = read_json(RESULT)
        cls.bundle = ROOT / cls.config['bundle']
        cls.manifest, cls.indexed = validate_bundle(cls.bundle)
        cls.selection = next(s for s in cls.config['selections'] if s['site_id'] == 'quoted-480k')

    def altered_manifest(self, change):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'evidence'
            shutil.copytree(self.bundle, bundle)
            manifest = read_json(bundle / 'manifest.json')
            change(manifest)
            (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            return validate_bundle(bundle)

    def test_real_480k_denominator_is_ten(self):
        projected = project_result(self.data, self.config, self.indexed)
        row = next(r for r in projected['comparison']['rows'] if r['id'] == 'quoted-480k')
        self.assertEqual((row['correct'], row['asked'], score(row)), (10, 10, '10/10'))

    def test_rescaled_denominator_rejected_against_copied_grader(self):
        def change(manifest):
            row = next(r for r in manifest['rows'] if r['id'] == self.selection['trial_id'])
            row['correct'] = row['asked'] = 24
        with self.assertRaisesRegex(ValueError, 'Grader .* mismatch'):
            self.altered_manifest(change)

    def test_void_status_cannot_be_erased(self):
        def change(manifest):
            next(r for r in manifest['rows'] if r['void'])['void'] = False
        with self.assertRaisesRegex(ValueError, 'Grader void mismatch'):
            self.altered_manifest(change)

    def test_void_seed_is_displayed_as_void(self):
        config = copy.deepcopy(self.config)
        selected = next(s for s in config['selections'] if s['site_id'] == 'read-improved-119k' and s['seed'] == 1)
        previous = next(r for r in self.manifest['rows'] if r['arm'] == 'B32ir' and r['seed'] == 1 and r['void'])
        selected.update(trial_id=previous['id'], source_record_id=previous['source_record_id'])
        projected = project_result(self.data, config, self.indexed)
        row = next(r for r in projected['comparison']['rows'] if r['id'] == 'read-improved-119k')
        trial = next(t for t in row['trials'] if t['seed'] == 1)
        self.assertEqual((trial['correct'], trial['asked']), (18, 18))
        self.assertEqual(score(trial), 'Void (rule breach)')

    def test_missing_completed_seed_cannot_fall_back_to_old_result(self):
        indexed = dict(self.indexed)
        del indexed[self.selection['trial_id']]
        with self.assertRaisesRegex(ValueError, 'absent from manifest'):
            project_result(self.data, self.config, indexed)

    def test_interrupted_trial_cannot_be_selected_as_result(self):
        indexed = copy.deepcopy(self.indexed)
        indexed[self.selection['trial_id']]['status'] = 'interrupted'
        with self.assertRaisesRegex(ValueError, 'not completed'):
            project_result(self.data, self.config, indexed)

    def test_refresh_cannot_silently_drop_failed_attempt(self):
        incoming = dict(self.indexed)
        failed = next(r for r in self.manifest['rows'] if r['void'])
        del incoming[failed['id']]
        with self.assertRaisesRegex(ValueError, 'drops an existing attempt'):
            preserve_history(self.indexed, incoming)

    def test_wrong_arm_or_seed_rejected(self):
        for field, bad_value in [('arm', 'B32ir'), ('seed', 1)]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                next(s for s in config['selections'] if s['site_id'] == 'quoted-480k')['expect'][field] = bad_value
                with self.assertRaisesRegex(ValueError, 'Selected .* mismatch'):
                    project_result(self.data, config, self.indexed)

    def test_corrupted_source_rejected_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'evidence'
            shutil.copytree(self.bundle, bundle)
            trial = self.indexed[self.selection['trial_id']]
            source = bundle / trial['sources']['verifier']['copy_path']
            source.write_bytes(source.read_bytes().replace(b'"correct": 10', b'"correct": 24', 1))
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                validate_bundle(bundle)

    def test_wrong_task_fingerprint_rejected(self):
        def change(manifest):
            manifest['rows'][0]['task_fingerprint'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Task fingerprint mismatch'):
            self.altered_manifest(change)

    def test_unselected_history_preserved_and_projection_stable(self):
        projected = project_result(self.data, self.config, self.indexed)
        selected = {s['site_id'] for s in self.config['selections']}
        for before, after in zip(self.data['comparison']['rows'], projected['comparison']['rows']):
            if before['id'] not in selected:
                self.assertEqual(before, after)
        self.assertEqual(projected, project_result(projected, self.config, self.indexed))


if __name__ == '__main__':
    unittest.main()
