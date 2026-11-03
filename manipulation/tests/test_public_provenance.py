"""Pure public output/environment contracts; no simulator, model or hardware query."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from manip import provenance as manipulation
from wm import provenance as world_model


class PublicProvenanceTests(unittest.TestCase):
    def test_explicit_output_path_needs_no_machine_specific_prefix(self):
        chosen = Path(__file__).resolve().parents[2]/'user-selected-output'
        for module in (manipulation, world_model):
            self.assertEqual(module.artifact_root(chosen), chosen.resolve())

    def test_git_metadata_output_is_rejected_case_insensitively(self):
        for module in (manipulation, world_model):
            for component in ('.git', '.GIT'):
                with self.assertRaises(ValueError):
                    module.artifact_root(Path(__file__).resolve().parent/component/'output')

    def identity(self,module,system,release,env=None,markers=()):
        def marker(path):
            return str(path).replace('\\','/') in markers
        with patch.dict(os.environ,env or {},clear=True), \
             patch.object(module.platform,'system',return_value=system), \
             patch.object(module.platform,'release',return_value=release), \
             patch.object(module.Path,'is_file',marker):
            return module.environment_identity()

    def test_detected_environment_has_actual_platform_or_marker_evidence(self):
        cases = [('Linux','6.0',{},(),'Linux'),
                 ('Darwin','24.0',{},(),'Darwin'),
                 ('Windows','11',{},(),'native Windows'),
                 ('Linux','6.0-microsoft-standard-WSL2',{},(),'WSL2'),
                 ('Linux','4.4-microsoft',{},(),'WSL'),
                 ('Linux','6.0',{'WSL_INTEROP':'present'},(),'WSL'),
                 ('Linux','6.0',{'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'self-hosted'},(),'GitHub Actions'),
                 ('Linux','6.0',{},('/.dockerenv',),'Docker/container')]
        for module in (manipulation, world_model):
            for system,release,env,markers,expected in cases:
                result=self.identity(module,system,release,env,markers)
                self.assertEqual(result['detected'],expected)
                self.assertEqual(result['label'],expected)
                self.assertEqual(result['label_source'],'detected')
                self.assertEqual(result['evidence']['platform_system'],system)
                self.assertEqual(result['evidence']['runner_environment'],env.get('RUNNER_ENVIRONMENT'))

    def test_user_label_does_not_replace_detected_evidence(self):
        for module in (manipulation, world_model):
            result=self.identity(module,'Linux','6.0',{'M3_ENVIRONMENT':'my-lab'})
            self.assertEqual(result['label'],'my-lab')
            self.assertEqual(result['label_source'],'user-provided M3_ENVIRONMENT')
            self.assertEqual(result['detected'],'Linux')


if __name__ == '__main__':
    unittest.main()
